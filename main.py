import os
from typing import Dict, List, Tuple, Optional, Union
import torch as t
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import torch.nn.functional as f
import fire
from termcolor import colored as c

REPO_ID = "unsloth/Llama-3.2-1B-Instruct"
BATCH_SIZE = 1
VOCAB_SIZE = 128_256
CONTEXT_LENGTH = 131_072
EMBEDDING_DIM = 2048
N_HEADS = 32
N_LAYERS = 16
HIDDEN_DIM = 8192
N_KV_GROUPS = 8
ROPE_BASE = 500_000.0
DTYPE = t.bfloat16
ROPE_FREQ = {
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_context_length": 8192,
}
HEAD_DIM = EMBEDDING_DIM // N_HEADS
GROUP_SIZE = N_HEADS // N_KV_GROUPS


def get_best_device() -> t.device:
    if t.cuda.is_available():
        return t.device("cuda")
    elif t.backends.mps.is_available():
        return t.device("mps")
    else:
        return t.device("cpu")


def precompute_rope_params(
    head_dim: int,
    theta_base: float,
    context_length: int,
    original_context_length: int,
    low_freq_factor: float,
    high_freq_factor: float,
    factor: float,
    device: t.device,
) -> Tuple[t.Tensor, t.Tensor]:
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    inv_freq = 1.0 / (
        theta_base ** (t.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
    )

    low_freq_wavelen = original_context_length / low_freq_factor
    high_freq_wavelen = original_context_length / high_freq_factor

    wavelen = 2 * t.pi / inv_freq
    inv_freq_llama = t.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)

    smooth_factor = (original_context_length / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (1 - smooth_factor) * (
        inv_freq / factor
    ) + smooth_factor * inv_freq

    is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
    inv_freq_llama = t.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    inv_freq = inv_freq_llama

    positions = t.arange(context_length)
    angles = positions[:, None] * inv_freq[None, :]
    angles = t.cat([angles, angles], dim=1)

    cos = t.cos(angles)
    sin = t.sin(angles)

    return cos.to(device), sin.to(device)


def compute_rope(x: t.Tensor, cos: t.Tensor, sin: t.Tensor) -> t.Tensor:
    _, _, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = t.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


def rescale_theta(
    theta_old: float, context_length_old: int, context_length_new: int
) -> float:
    scaling_factor = context_length_new / context_length_old
    theta_new = theta_old * scaling_factor
    return theta_new


class Llama:
    device: t.device
    tokenizer: AutoTokenizer
    tensors: Dict[str, t.Tensor]  # model tensors
    cos: t.Tensor  # (context_length, head_dim)
    sin: t.Tensor  # (context_length, head_dim)
    kv_cache_i: int  # count of tokens in kv_cache
    k_cache: List[
        t.Tensor
    ]  # for each layer: (b, num_kv_groups, context_length, head_dim)
    v_cache: List[
        t.Tensor
    ]  # for each layer: (b, num_kv_groups, context_length, head_dim)
    last_logits: Optional[t.Tensor] = None  # (b, 1, vocab_size)
    context_length: int

    def text_to_tokens(self, text: str) -> t.Tensor:
        return (
            t.tensor(self.tokenizer.encode(text, add_special_tokens=False))
            .to(self.device)
            .view(1, -1)
        )

    def tokens_to_text(self, token_ids: t.Tensor) -> str:
        return self.tokenizer.decode(token_ids.view(-1).tolist())

    def encode_header(self, role: str) -> t.Tensor:
        tokens = []
        tokens.append(self.tokenizer.get_added_vocab()["<|start_header_id|>"])
        tokens.extend(self.text_to_tokens(role))
        tokens.append(self.tokenizer.get_added_vocab()["<|end_header_id|>"])
        tokens.extend(self.text_to_tokens("\n\n"))
        return t.tensor(tokens).view(1, -1).to(self.device)

    def ffn(self, x: t.Tensor, layer_i: int) -> t.Tensor:
        x_fc1 = f.linear(
            x, self.tensors[f"model.layers.{layer_i}.mlp.gate_proj.weight"]
        )
        x_fc2 = f.linear(x, self.tensors[f"model.layers.{layer_i}.mlp.up_proj.weight"])
        x = f.silu(x_fc1) * x_fc2
        return f.linear(x, self.tensors[f"model.layers.{layer_i}.mlp.down_proj.weight"])

    def attention(self, x: t.Tensor, layer_i: int) -> t.Tensor:
        b, num_tokens, _ = x.shape

        queries = f.linear(
            x, self.tensors[f"model.layers.{layer_i}.self_attn.q_proj.weight"]
        )  # (b, num_tokens, d_out)
        keys = f.linear(
            x, self.tensors[f"model.layers.{layer_i}.self_attn.k_proj.weight"]
        )  # (b, num_tokens, num_kv_groups * head_dim)
        values = f.linear(
            x, self.tensors[f"model.layers.{layer_i}.self_attn.v_proj.weight"]
        )  # (b, num_tokens, num_kv_groups * head_dim)

        # reshape for easier manipulation
        queries = queries.view(b, num_tokens, N_HEADS, HEAD_DIM).transpose(
            1, 2
        )  # (b, num_heads, num_tokens, head_dim)
        keys = keys.view(b, num_tokens, N_KV_GROUPS, HEAD_DIM).transpose(
            1, 2
        )  # (b, num_kv_groups, num_tokens, head_dim)
        values = values.view(b, num_tokens, N_KV_GROUPS, HEAD_DIM).transpose(
            1, 2
        )  # (b, num_kv_groups, num_tokens, head_dim)

        # Apply RoPE before caching
        queries = compute_rope(
            queries,
            self.cos[self.kv_cache_i : self.kv_cache_i + num_tokens],
            self.sin[self.kv_cache_i : self.kv_cache_i + num_tokens],
        )
        keys = compute_rope(
            keys,
            self.cos[self.kv_cache_i : self.kv_cache_i + num_tokens],
            self.sin[self.kv_cache_i : self.kv_cache_i + num_tokens],
        )

        # add keys and values to cache
        self.k_cache[layer_i][
            :, :, self.kv_cache_i : self.kv_cache_i + num_tokens
        ] = keys
        self.v_cache[layer_i][
            :, :, self.kv_cache_i : self.kv_cache_i + num_tokens
        ] = values

        # get keys and values from cache
        keys = self.k_cache[layer_i][
            :, :, : self.kv_cache_i + num_tokens
        ]  # (b, num_heads, kv_cache_i + num_tokens, head_dim)
        values = self.v_cache[layer_i][
            :, :, : self.kv_cache_i + num_tokens
        ]  # (b, num_heads, kv_cache_i + num_tokens, head_dim)

        # expand keys and values to match the number of heads
        keys = keys.repeat_interleave(
            GROUP_SIZE, dim=1
        )  # (b, num_heads, kv_cache_i + num_tokens, head_dim)
        values = values.repeat_interleave(
            GROUP_SIZE, dim=1
        )  # (b, num_heads, kv_cache_i + num_tokens, head_dim)

        attn_scores = queries @ keys.transpose(
            2, 3
        )  # (b, num_heads, num_tokens, kv_cache_i + num_tokens)

        mask = t.triu(
            t.ones(num_tokens, self.kv_cache_i + num_tokens),
            diagonal=self.kv_cache_i + 1,
        ).to(
            self.device
        )  # (num_tokens, kv_cache_i + num_tokens)
        mask = mask.bool()
        attn_scores.masked_fill_(mask, -t.inf)

        attn_weights = t.softmax(
            attn_scores / HEAD_DIM**0.5, dim=-1
        )  # (b, num_heads, num_tokens, kv_cache_i + num_tokens)

        context_vec = (attn_weights @ values).transpose(
            1, 2
        )  # (b, num_tokens, num_heads, head_dim)

        context_vec = context_vec.reshape(
            b, num_tokens, EMBEDDING_DIM
        )  # (b, num_tokens, d_out)

        context_vec = f.linear(
            context_vec, self.tensors[f"model.layers.{layer_i}.self_attn.o_proj.weight"]
        )  # (b, num_tokens, d_out)

        return context_vec

    def transformer_block(self, x: t.Tensor, layer_i: int) -> t.Tensor:
        x += self.attention(
            f.rms_norm(
                x,
                (EMBEDDING_DIM,),
                self.tensors[f"model.layers.{layer_i}.input_layernorm.weight"],
                eps=1e-5,
            ),
            layer_i,
        )
        x += self.ffn(
            f.rms_norm(
                x,
                (EMBEDDING_DIM,),
                self.tensors[f"model.layers.{layer_i}.post_attention_layernorm.weight"],
                eps=1e-5,
            ),
            layer_i,
        )

        return x

    def eval(self, tokens: t.Tensor, text_color: str = "white") -> None:
        print(c(self.tokens_to_text(tokens), text_color), end="", flush=False)

        if self.kv_cache_i + tokens.shape[1] > self.context_length:
            raise ValueError(
                f"Too many tokens: {self.kv_cache_i + tokens.shape[1]}, max: {self.context_length}, increase context_length"
            )

        x = f.embedding(
            tokens, self.tensors["model.embed_tokens.weight"], padding_idx=0
        )

        for i in range(N_LAYERS):
            x = self.transformer_block(x, i)
        self.kv_cache_i += tokens.shape[1]

        x = f.rms_norm(
            x,
            (EMBEDDING_DIM,),
            self.tensors["model.norm.weight"],
            eps=1e-5,
        )

        # we only care about the logits for the last token
        x = x[:, -1, :].unsqueeze(1)

        logits = f.linear(x, self.tensors["model.embed_tokens.weight"])
        self.last_logits = logits

    def generate(self, max_new_tokens: int, temperature: float, top_k: int) -> None:
        eos_id = self.tokenizer.get_added_vocab()["<|eot_id|>"]

        if self.last_logits is None:
            raise ValueError("Please run eval() before generate()")

        for _ in range(max_new_tokens):
            logits = self.last_logits[:, -1, :]

            top_logits, _ = t.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = t.where(
                logits < min_val, t.tensor(float("-inf")).to(logits.device), logits
            )

            if temperature > 0.0:
                logits = logits / temperature
                probs = t.softmax(logits, dim=-1)
                next_token = t.multinomial(probs, num_samples=1)
            else:
                next_token = t.argmax(logits, dim=-1, keepdim=True)

            if next_token == eos_id:
                break

            with t.no_grad():
                self.eval(next_token)

    def __init__(
        self,
        context_length: int,
        device: Union[str, t.device],
        seed: int,
    ):
        t.manual_seed(seed)

        if isinstance(device, str) and device in ["cuda", "cpu"]:
            self.device = t.device(device)
        else:
            self.device = get_best_device()
        print("Using device:", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
        weights_file = hf_hub_download(
            repo_id=REPO_ID,
            filename="model.safetensors",
            local_dir=f"./Llama-3.2-1B-Instruct",
        )
        self.tensors = load_file(weights_file)
        for k, v in self.tensors.items():
            self.tensors[k] = v.to(self.device)

        self.context_length = context_length
        scaled_rope_base = rescale_theta(
            ROPE_BASE,
            CONTEXT_LENGTH,
            context_length,
        )
        print("New RoPE theta:", scaled_rope_base)

        cos, sin = precompute_rope_params(
            HEAD_DIM,
            scaled_rope_base,
            context_length,
            ROPE_FREQ["original_context_length"],
            ROPE_FREQ["low_freq_factor"],
            ROPE_FREQ["high_freq_factor"],
            ROPE_FREQ["factor"],
            self.device,
        )
        self.cos = cos
        self.sin = sin

        self.kv_cache_i = 0
        self.k_cache = [
            t.zeros(
                BATCH_SIZE,
                N_KV_GROUPS,
                context_length,
                HEAD_DIM,
                dtype=DTYPE,
            ).to(self.device)
            for _ in range(N_LAYERS)
        ]
        self.v_cache = [
            t.zeros(
                BATCH_SIZE,
                N_KV_GROUPS,
                context_length,
                HEAD_DIM,
                dtype=DTYPE,
            ).to(self.device)
            for _ in range(N_LAYERS)
        ]

        print()


def terminal_line() -> None:
    print("-" * os.get_terminal_size().columns)


def main(
    context_length: int = 8192,
    seed: int = 123,
    max_new_tokens: int = 1000,
    temperature: float = 0.0,
    top_k: int = 1,
    force_cpu: bool = False,
) -> None:
    print()
    if force_cpu:
        device = "cpu"
    else:
        device = get_best_device()

    llama = Llama(
        context_length,
        device,
        seed,
    )

    eot_tensor = (
        t.tensor(llama.tokenizer.get_added_vocab()["<|eot_id|>"])
        .view(1, 1)
        .to(llama.device)
    )

    llama.eval(llama.encode_header("system"), "yellow")
    llama.eval(llama.text_to_tokens("You are a helpful assistant."))
    llama.eval(eot_tensor, "red")

    while True:
        print()
        terminal_line()
        text = input("You: ")
        if text == "exit":
            break
        terminal_line()

        llama.eval(llama.encode_header("user"), "cyan")
        llama.eval(llama.text_to_tokens(text))
        llama.eval(eot_tensor, "red")

        llama.eval(llama.encode_header("assistant"), "magenta")
        llama.generate(max_new_tokens, temperature, top_k)
        llama.eval(eot_tensor, "red")


if __name__ == "__main__":
    fire.Fire(main)
