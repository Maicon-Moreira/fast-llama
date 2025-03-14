# Tiny Llama

Blazingly fast minimal implementation of Llama 3.2 1B in PyTorch (~400 lines).

Uses [unsloth/Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) tokenizer and weights.

## Install

```bash
pip install torch transformers huggingface_hub safetensors fire termcolor
```

## Usage

```bash
python main.py --context_length 8192 \      # optional
               --seed 123 \                 # optional
               --max_new_tokens 1000 \      # optional
               --temperature 0.0 \          # optional, 0 for deterministic output
               --top_k 1 \                  # optional
               --force_cpu                  # optional, use CPU even if GPU is available
```

Example:

```bash
python main.py
```

## Features

- KV caching for efficient inference
- RoPE position embeddings with scaling
- Interactive chat interface in the terminal
- Support for GPU, MPS (Mac), and CPU inference
