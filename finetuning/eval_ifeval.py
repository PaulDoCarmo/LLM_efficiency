#!/usr/bin/env python
"""Évalue un modèle sur IFEval, avec ou sans adaptateur LoRA.

Sert à mesurer le gain du finetuning de finetune_lora.py : lance-le une fois
sans --adapter (base de référence) et une fois avec, et compare.

Le modèle est chargé en 4bit NF4 par défaut, dans les mêmes conditions qu'à
l'entraînement. Comme le finetuning utilise le chat template, l'évaluation
l'applique aussi dès qu'un adaptateur est chargé (--no-chat-template pour
forcer les prompts bruts).

Usage:
    python finetuning/eval_ifeval.py                                  # base, référence
    python finetuning/eval_ifeval.py --adapter finetuning/out/...     # finetuné
    python finetuning/eval_ifeval.py --adapter ... --limit 40         # aperçu rapide
"""
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

IFEVAL_METRICS = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]


def extract_metric(metrics, name):
    """lm-eval suffixe les clés du nom du filtre (ex: 'prompt_level_strict_acc,none')."""
    for k, v in metrics.items():
        if k.split(",")[0] == name:
            return v
    return None


def load_model(model_name, adapter, four_bit):
    quant_cfg = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if four_bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument(
        "--adapter",
        default=None,
        help="Dossier de l'adaptateur LoRA produit par finetune_lora.py. "
        "Omis = évalue le modèle de base (référence).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite le nombre d'exemples IFEval (défaut: tous, 541 prompts).",
    )
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument(
        "--no-4bit",
        dest="four_bit",
        action="store_false",
        help="Charge en bf16 au lieu de 4bit NF4.",
    )
    ap.add_argument(
        "--chat-template",
        dest="chat_template",
        action="store_true",
        help="Force l'application du chat template. Utile pour évaluer le modèle "
        "de base dans les mêmes conditions de formatage que le finetuné.",
    )
    ap.add_argument(
        "--no-chat-template",
        dest="chat_template",
        action="store_false",
        help="Force l'envoi des prompts IFEval bruts. Par défaut le template est "
        "appliqué dès qu'un adaptateur est chargé (cohérence avec l'entraînement).",
    )
    ap.add_argument("--out", help="Fichier JSON de résultats. Défaut: results/ifeval_<ts>.json")
    ap.set_defaults(four_bit=True, chat_template=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA requis.")

    # Par défaut : template appliqué si (et seulement si) on évalue un finetuné,
    # puisque c'est le format vu à l'entraînement.
    chat_template = args.chat_template
    if chat_template is None:
        chat_template = args.adapter is not None

    print(f"modèle  = {args.model}")
    print(f"adapter = {args.adapter or '(aucun — base de référence)'}")
    print(f"4bit    = {args.four_bit}   chat_template = {chat_template}")

    # Le tokenizer de l'adaptateur peut différer (tokens spéciaux ajoutés).
    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, args.adapter, args.four_bit)

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    logging.getLogger("lm-eval").setLevel(logging.WARNING)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size)
    out = simple_evaluate(
        model=lm,
        tasks=["ifeval"],
        limit=args.limit,
        apply_chat_template=chat_template,
        bootstrap_iters=0,
    )
    metrics = out["results"]["ifeval"]

    print("\nIFEval")
    print("-" * 34)
    for name in IFEVAL_METRICS:
        value = extract_metric(metrics, name)
        print(f"{name:26} {value * 100:6.2f}%" if value is not None else f"{name:26}    n/a")

    out_path = Path(args.out) if args.out else Path("results") / (
        f"ifeval_{'lora' if args.adapter else 'base'}_{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "adapter": args.adapter,
                "four_bit": args.four_bit,
                "chat_template": chat_template,
                "limit": args.limit,
                "batch_size": args.batch_size,
                "metrics": metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nrésultats écrits dans {out_path}")


if __name__ == "__main__":
    main()
