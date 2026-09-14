#!/usr/bin/env python3
"""Baseline ResNet50 sur le val set ImageNet : accuracy et energie GPU reelle,
mesuree via le compteur materiel NVML (nvmlDeviceGetTotalEnergyConsumption),
avant une etude de pruning.

Dataset attendu (lecture seule, rien n'est jamais ecrit sous /raid) :
- --data-root/train/ : ImageFolder classique, utilise uniquement pour deriver
  l'ordre trie des classes (class_to_idx), pas pour l'entrainement.
- --data-root/val/ : 50000 JPEG a plat (pas de sous-dossiers de classe) ; les
  labels sont lus depuis --val-csv (colonnes ImageId, PredictionString).

Tout ce que ce script ecrit (JSON de resultats) va sous /home/fmr2026, jamais
sous /raid.

La logique de mesure (Dataset, evaluate(), selection GPU) vit dans
cli_common.py (stdlib) et resnet50_eval_lib.py (torch), partages avec
resnet50_prune_finetune.py pour garantir une comparaison baseline/pruned
strictement apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cli_common import add_shared_eval_args, bootstrap_gpu_selection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shared_eval_args(p)
    p.add_argument("--out", default=None,
                   help="Chemin du JSON de sortie (defaut : pruning/runs/<horodatage>.json, "
                        "sous /home/fmr2026). Volontairement PAS pruning/results/ : ce nom est "
                        "couvert par le .gitignore du repo (convention des autres benchmarks) ; "
                        "pruning/runs/ reste versionne pour pouvoir push/pull les resultats "
                        "depuis le cluster.")
    return p.parse_args()


_ARGS = parse_args()
bootstrap_gpu_selection(_ARGS.gpu)

# Import differe : torch ne doit voir qu'un seul GPU (fixe ci-dessus). NVML,
# lui, enumere toujours TOUS les GPU physiques quel que soit CUDA_VISIBLE_DEVICES
# (voir resnet50_eval_lib.init_gpu : le handle NVML est pris sur l'index
# physique --gpu, pas sur 0).
from torchvision.models import ResNet50_Weights, resnet50  # noqa: E402

from resnet50_eval_lib import (  # noqa: E402
    build_class_to_idx,
    build_val_dataset,
    evaluate,
    init_gpu,
    print_validity_check,
)


def main() -> None:
    args = _ARGS

    handle, gpu_name, device = init_gpu(args.gpu)

    weights = ResNet50_Weights[args.weights]
    transform = weights.transforms()

    data_root = Path(args.data_root)
    class_to_idx = build_class_to_idx(data_root / "train")
    print(f"{len(class_to_idx)} classes indexees depuis {data_root / 'train'} (ordre trie des synsets)")

    dataset = build_val_dataset(data_root, args.val_csv, class_to_idx, transform, args.n_images)

    model = resnet50(weights=weights).to(device).eval()

    results = evaluate(model, device, handle, gpu_name, dataset, args)

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path(__file__).resolve().parent / "runs"
        ts = time.strftime("%Y%m%dT%H%M%S")
        out_path = out_dir / f"resnet50_baseline_{args.weights}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"Resultats ecrits dans {out_path}")

    print_validity_check(results["top1_acc"], args.weights)


if __name__ == "__main__":
    main()
