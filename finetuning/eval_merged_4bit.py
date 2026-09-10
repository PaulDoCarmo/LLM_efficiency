#!/usr/bin/env python
"""Fusionne l'adaptateur LoRA dans la base, requantifie en 4 bits, évalue sur IFEval.

Le pipeline est en trois temps :

    base bf16  +  adaptateur LoRA
        ↓  merge_and_unload()          (obligatoirement en pleine précision)
    modèle fusionné bf16 (~3 Go sur disque, temporaire)
        ↓  BitsAndBytesConfig(load_in_4bit=True)
    modèle fusionné 4 bits  →  IFEval

La fusion DOIT se faire en bf16 : additionner `B@A` dans des poids déjà
quantifiés n'a pas de sens (il faudrait déquantifier, additionner, requantifier
— ce que fait exactement ce script, mais en une passe propre).

L'évaluation réutilise `run_ifeval` de benchmark.py (import direct, pas de
copie) pour que le score soit directement comparable aux lignes fp16/bf16/
int8/4bit du tableau de benchmark.py.

Usage:
    # test du pipeline AVANT la fin de l'entraînement (adaptateur non entraîné)
    python finetuning/eval_merged_4bit.py --dummy --limit 5

    # une fois l'entraînement fini
    python finetuning/eval_merged_4bit.py \
        --adapter finetuning/out/qwen2.5-1.5b-qlora-ifeval --limit 100
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Réutilise le monitoring de benchmark.py (IFEval, débit, énergie, VRAM)
# plutôt que de le dupliquer : les chiffres restent comparables au tableau
# des variantes de quantification.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark import (  # noqa: E402
    _extract_metric,
    energy_output_dir,
    gpu_list_from_env,
    measure_loaded_model,
    print_table,
)
from finetune_lora import QWEN_LORA_TARGETS  # noqa: E402

IFEVAL_METRICS = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]


def quant_config(quant_type):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def attach_adapter(model, adapter_dir, dummy, lora_r, lora_alpha):
    """Applique l'adaptateur LoRA (ou en fabrique un non entraîné en mode dummy).

    En --dummy, la matrice B de LoRA est initialisée à zéro, donc B@A = 0 et la
    fusion est un no-op mathématique : le score doit reproduire celui de la base
    quantifiée. C'est ce qui rend ce mode utile comme vérification, et pas
    seulement comme test anti-crash.
    """
    if dummy:
        from peft import LoraConfig, get_peft_model

        return get_peft_model(
            model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=QWEN_LORA_TARGETS,
            ),
        )
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_dir)


def merge_into_quantized(base_model, adapter_dir, dummy, lora_r, lora_alpha, quant_type):
    """Fusion FIDÈLE aux conditions d'entraînement : quantize(dequant(W_nf4) + ΔW).

    L'adaptateur a été entraîné contre `dequant(W_nf4)`, pas contre `W` : une
    partie de ce que ΔW a appris est la compensation de l'erreur de
    quantification. On recharge donc la base avec la MÊME config 4 bits qu'à
    l'entraînement, et `merge_and_unload()` de PEFT déquantifie chaque couche,
    additionne ΔW, puis requantifie sur place. Pas de détour par le disque.
    """
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config(quant_type),
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = attach_adapter(model, adapter_dir, dummy, lora_r, lora_alpha)
    merged = model.merge_and_unload()
    merged.eval()
    return merged


def merge_into_bf16(base_model, adapter_dir, merged_dir, dummy, lora_r, lora_alpha, quant_type):
    """Fusion dans les poids bf16 d'origine : quantize(W + ΔW).

    Plus fidèle aux poids originaux, mais PAS aux conditions d'entraînement —
    l'adaptateur corrigeait une base quantifiée. Gardé pour pouvoir comparer les
    deux stratégies. Passe par le disque (~3 Go temporaires) car la
    requantification se fait au rechargement.
    """
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map={"": 0}
    )
    model = attach_adapter(model, adapter_dir, dummy, lora_r, lora_alpha)
    merged = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    del model, merged
    torch.cuda.empty_cache()

    requantized = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        quantization_config=quant_config(quant_type),
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    requantized.eval()
    return requantized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument(
        "--adapter",
        default=None,
        help="Dossier de l'adaptateur LoRA produit par finetune_lora.py. "
        "Requis sauf en mode --dummy.",
    )
    ap.add_argument(
        "--dummy",
        action="store_true",
        help="Fabrique un adaptateur LoRA non entraîné au lieu d'en charger un. "
        "Sert à tester le pipeline complet avant la fin de l'entraînement : "
        "la fusion est un no-op, donc le score doit égaler celui de la base 4bit.",
    )
    ap.add_argument(
        "--quant-type",
        default="nf4",
        choices=["nf4", "fp4"],
        help="Quantification 4 bits du modèle fusionné. 'nf4' (défaut) = celle de "
        "l'entraînement QLoRA. 'fp4' = celle de la variante '4bit' de benchmark.py, "
        "à utiliser pour une comparaison stricte avec cette ligne du tableau.",
    )
    ap.add_argument(
        "--merge-into",
        default="quantized",
        choices=["quantized", "bf16"],
        help="Cible de la fusion. 'quantized' (défaut) = quantize(dequant(W_nf4) + ΔW), "
        "fidèle aux conditions d'entraînement puisque l'adaptateur a appris contre "
        "la base quantifiée. 'bf16' = quantize(W + ΔW), fusion dans les poids "
        "d'origine (passe par le disque, ~3 Go temporaires).",
    )
    ap.add_argument(
        "--merged-dir",
        default="finetuning/out/merged",
        help="Dossier temporaire du modèle fusionné. Utilisé par --merge-into bf16 seulement.",
    )
    ap.add_argument(
        "--drop-merged",
        action="store_true",
        help="Supprime le modèle fusionné bf16 (~3 Go) après l'évaluation.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Nb d'exemples IFEval (défaut: 541).")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16, help="Mode --dummy uniquement.")
    ap.add_argument("--lora-alpha", type=int, default=32, help="Mode --dummy uniquement.")
    ap.add_argument(
        "--gen-tokens",
        type=int,
        default=128,
        help="Longueur de génération pour la mesure de débit (comme benchmark.py).",
    )
    ap.add_argument(
        "--chat-template",
        action="store_true",
        help="Applique le chat template aux prompts IFEval. À activer si l'adaptateur "
        "a été entraîné au format chat (c'est le cas de finetune_lora.py) ; laissé "
        "désactivé par défaut pour rester identique à benchmark.py.",
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
    ap.add_argument("--out", help="Fichier JSON de résultats.")
    ap.set_defaults(measure_energy=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA requis (bitsandbytes ne quantifie que sur GPU).")
    if not args.dummy and not args.adapter:
        raise SystemExit("--adapter est requis (ou utilise --dummy pour tester le pipeline).")
    if not args.dummy and not Path(args.adapter).is_dir():
        raise SystemExit(f"adaptateur introuvable : {args.adapter}")

    merged_dir = Path(args.merged_dir)
    # Le modèle tourne sur le premier GPU visible ; nvidia-smi, lui, voit
    # toujours l'index physique (cf. energy_measurement/README.md).
    gpus = gpu_list_from_env()
    energy_gpu_index = (
        args.energy_gpu_index
        if args.energy_gpu_index is not None
        else (int(gpus[0]) if gpus else 0)
    )

    print(f"base       = {args.base}")
    print(f"adaptateur = {'(dummy, non entraîné)' if args.dummy else args.adapter}")
    print(f"quant      = {args.quant_type}   fusion -> {args.merge_into}   batch = {args.batch_size}")
    print(f"énergie    = {args.measure_energy} (GPU physique {energy_gpu_index})")

    tok = AutoTokenizer.from_pretrained(args.base if args.dummy else args.adapter)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if args.merge_into == "quantized":
        print("\n[1/2] chargement 4 bits + fusion de l'adaptateur en place...")
        model = merge_into_quantized(
            args.base, args.adapter, args.dummy, args.lora_r, args.lora_alpha, args.quant_type
        )
    else:
        print("\n[1/2] fusion dans les poids bf16 puis requantification...")
        model = merge_into_bf16(
            args.base,
            args.adapter,
            merged_dir,
            args.dummy,
            args.lora_r,
            args.lora_alpha,
            args.quant_type,
        )
    print("[2/2] débit + IFEval + énergie (monitoring de benchmark.py)...")
    label = "dummy" if args.dummy else "lora"
    variant = f"merged4bit-{label}-{args.quant_type}"
    energy_dir = (
        Path(args.energy_out) / label
        if args.energy_out
        else energy_output_dir(args.base, variant)
    )
    result = measure_loaded_model(
        model,
        tok,
        gen_tokens=args.gen_tokens,
        ifeval=True,
        ifeval_limit=args.limit,
        ifeval_batch_size=args.batch_size,
        ifeval_chat_template=args.chat_template,
        measure_energy=args.measure_energy,
        energy_gpu_index=energy_gpu_index,
        energy_dir=energy_dir,
        energy_metadata={
            "model": args.base,
            "adapter": args.adapter,
            "variant": variant,
            "gen_tokens": args.gen_tokens,
            "merge_into": args.merge_into,
        },
    )
    result["variant"] = label
    result["gpu"] = energy_gpu_index
    metrics = result.get("ifeval", {})

    # Même tableau que benchmark.py (fonction importée), donc colonnes alignées.
    table_lines = print_table([result], ifeval=True)

    print("\nIFEval (détail)")
    print("-" * 34)
    for name in IFEVAL_METRICS:
        value = _extract_metric(metrics, name)
        print(f"{name:26} {value * 100:6.2f}%" if value is not None else f"{name:26}    n/a")

    out_path = Path(args.out) if args.out else Path("results") / (
        f"ifeval_merged4bit_{'dummy' if args.dummy else 'lora'}_"
        f"{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base": args.base,
                "adapter": None if args.dummy else args.adapter,
                "dummy": args.dummy,
                "quant_type": args.quant_type,
                "merge_into": args.merge_into,
                "limit": args.limit,
                "batch_size": args.batch_size,
                "gen_tokens": args.gen_tokens,
                "chat_template": args.chat_template,
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

    if args.merge_into == "bf16":
        if args.drop_merged:
            shutil.rmtree(merged_dir, ignore_errors=True)
            print(f"modèle fusionné supprimé ({merged_dir})")
        else:
            print(f"modèle fusionné conservé dans {merged_dir} (--drop-merged pour le supprimer)")


if __name__ == "__main__":
    main()
