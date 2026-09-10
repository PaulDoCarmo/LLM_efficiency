#!/usr/bin/env python
"""Évalue un modèle sur IFEval, avec ou sans adaptateur LoRA.

Sert à mesurer le gain du finetuning de finetune_lora.py : lance-le une fois
sans --adapter (base de référence) et une fois avec, et compare.

Le monitoring est **exactement celui de benchmark.py** : on importe son
`measure_loaded_model`, donc même chauffe hors mesure, même fenêtre
EnergyMeasurement, mêmes clés de résultat (tok_s, energy_wh, mean_power_w,
vram_gb, ...) et même tableau d'affichage.

Le modèle est chargé en 4bit NF4 par défaut, dans les mêmes conditions qu'à
l'entraînement. Comme le finetuning utilise le chat template, l'évaluation
l'applique aussi dès qu'un adaptateur est chargé (--no-chat-template pour
forcer les prompts bruts).

Usage:
    python finetuning/eval_ifeval.py --limit 100 --chat-template        # référence
    python finetuning/eval_ifeval.py --limit 100 --adapter finetuning/out/...
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Monitoring partagé avec benchmark.py : importé, pas recopié.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark import (  # noqa: E402
    _extract_metric,
    energy_output_dir,
    gpu_list_from_env,
    measure_loaded_model,
    print_table,
)

IFEVAL_METRICS = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]


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
        "--gen-tokens",
        type=int,
        default=128,
        help="Longueur de génération pour la mesure de débit (comme benchmark.py).",
    )
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
    ap.add_argument(
        "--no-energy",
        dest="measure_energy",
        action="store_false",
        help="Désactive la mesure d'énergie (activée par défaut, comme benchmark.py).",
    )
    ap.add_argument(
        "--energy-gpu-index",
        type=int,
        default=None,
        help="Index PHYSIQUE nvidia-smi du GPU à surveiller. Défaut: premier de "
        "CUDA_VISIBLE_DEVICES.",
    )
    ap.add_argument(
        "--energy-out",
        help="Dossier racine des traces d'énergie. Défaut: results/energy/<model>/<variant>/.",
    )
    ap.add_argument("--out", help="Fichier JSON de résultats. Défaut: results/ifeval_<ts>.json")
    ap.set_defaults(four_bit=True, chat_template=None, measure_energy=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA requis.")
    if args.adapter and not Path(args.adapter).is_dir():
        raise SystemExit(f"adaptateur introuvable : {args.adapter}")

    # Par défaut : template appliqué si (et seulement si) on évalue un finetuné,
    # puisque c'est le format vu à l'entraînement.
    chat_template = args.chat_template
    if chat_template is None:
        chat_template = args.adapter is not None

    # Le modèle tourne sur le premier GPU visible ; nvidia-smi, lui, voit
    # toujours l'index physique (cf. energy_measurement/README.md).
    gpus = gpu_list_from_env()
    energy_gpu_index = (
        args.energy_gpu_index
        if args.energy_gpu_index is not None
        else (int(gpus[0]) if gpus else 0)
    )

    label = "lora" if args.adapter else "base"
    variant = f"ifeval-{label}-{'4bit' if args.four_bit else 'bf16'}"

    print(f"modèle  = {args.model}")
    print(f"adapter = {args.adapter or '(aucun — base de référence)'}")
    print(f"4bit    = {args.four_bit}   chat_template = {chat_template}")
    print(f"énergie = {args.measure_energy} (GPU physique {energy_gpu_index})")

    # Le tokenizer de l'adaptateur peut différer (tokens spéciaux ajoutés).
    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = load_model(args.model, args.adapter, args.four_bit)

    energy_dir = (
        Path(args.energy_out) / label
        if args.energy_out
        else energy_output_dir(args.model, variant)
    )
    result = measure_loaded_model(
        model,
        tok,
        gen_tokens=args.gen_tokens,
        ifeval=True,
        ifeval_limit=args.limit,
        ifeval_batch_size=args.batch_size,
        ifeval_chat_template=chat_template,
        measure_energy=args.measure_energy,
        energy_gpu_index=energy_gpu_index,
        energy_dir=energy_dir,
        energy_metadata={
            "model": args.model,
            "adapter": args.adapter,
            "variant": variant,
            "gen_tokens": args.gen_tokens,
            "chat_template": chat_template,
        },
    )
    result["variant"] = label
    result["gpu"] = energy_gpu_index

    # Même tableau que benchmark.py (fonction importée), donc colonnes alignées.
    table_lines = print_table([result], ifeval=True)

    print("\nIFEval (détail)")
    print("-" * 34)
    for name in IFEVAL_METRICS:
        value = _extract_metric(result.get("ifeval", {}), name)
        print(f"{name:26} {value * 100:6.2f}%" if value is not None else f"{name:26}    n/a")

    out_path = (
        Path(args.out)
        if args.out
        else Path("results") / f"ifeval_{label}_{datetime.now():%Y%m%d-%H%M%S}.json"
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
                "gen_tokens": args.gen_tokens,
                "measure_energy": args.measure_energy,
                "gpu": energy_gpu_index,
                "result": result,
                "table": "\n".join(table_lines),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nrésultats écrits dans {out_path}")


if __name__ == "__main__":
    main()
