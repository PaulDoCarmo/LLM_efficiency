#!/usr/bin/env python
"""Compare un LLM en fp32 / fp16 / int8 / 4bit : perplexité, VRAM, débit.

Perplexité calculée sur WikiText-2 (test). VRAM = pic alloué. tok/s = génération
greedy de 128 tokens. Pensé pour un GPU Ampere+ (testé sur A100-40GB).

Usage:
    python benchmark.py --model Qwen/Qwen2.5-1.5B
    python benchmark.py --model meta-llama/Llama-3.2-1B --variants fp16 4bit --max-tokens 4000
"""
import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_configs(selected):
    all_cfg = {
        "fp32": dict(torch_dtype=torch.float32),
        "fp16": dict(torch_dtype=torch.float16),
        "int8": dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True)),
        "4bit": dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        ),
    }
    return {k: all_cfg[k] for k in selected}


@torch.no_grad()
def perplexity(model, enc, stride=512, max_len=2048):
    """Perplexité par fenêtre glissante (approche standard HF)."""
    nlls = []
    for i in range(0, enc.input_ids.size(1), stride):
        ids = enc.input_ids[:, i : i + max_len].to(model.device)
        tgt = ids.clone()
        tgt[:, :-stride] = -100  # ne scorer que les nouveaux tokens
        nlls.append(model(ids, labels=tgt).loss)
    return torch.exp(torch.stack(nlls).mean()).item()


def throughput(model, tok, n=128):
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(model.device)
    torch.cuda.synchronize()
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=n, do_sample=False)
    torch.cuda.synchronize()
    return (out.size(1) - ids.size(1)) / (time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["fp32", "fp16", "int8", "4bit"],
        choices=["fp32", "fp16", "int8", "4bit"],
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Tronque le texte d'éval à N tokens (0 = tout WikiText-2, ~330k, lent).",
    )
    ap.add_argument("--gen-tokens", type=int, default=128)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA requis (bitsandbytes ne quantifie que sur GPU).")

    tok = AutoTokenizer.from_pretrained(args.model)
    text = "\n\n".join(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    )
    enc = tok(text, return_tensors="pt")
    if args.max_tokens:
        enc.input_ids = enc.input_ids[:, : args.max_tokens]

    print(f"\nmodel = {args.model}   device = {torch.cuda.get_device_name()}")
    print(f"{'variant':8} {'ppl':>9} {'VRAM_GB':>9} {'tok/s':>8}")
    print("-" * 38)

    for name, cfg in build_configs(args.variants).items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="cuda", **cfg
        )
        model.eval()

        tps = throughput(model, tok, args.gen_tokens)
        ppl = perplexity(model, enc)
        vram = torch.cuda.max_memory_allocated() / 1e9

        print(f"{name:8} {ppl:9.3f} {vram:9.2f} {tps:8.1f}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()