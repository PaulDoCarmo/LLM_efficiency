#!/usr/bin/env python3
"""Re-evaluation fiable (50000 images du val set ImageNet, pas un sous-
ensemble) du modele baseline ResNet50 pre-entraine vs un checkpoint
pruned+finetune deja produit par resnet50_prune_finetune.py.

Les JSON existants sous pruning/runs/ (issus de resnet50_prune_finetune.py)
ont ete mesures avec --n-images 5000 : pas assez d'images pour des chiffres
d'accuracy/energie fiables. Ce script ne re-prune ni ne re-finetune rien : il
recharge le checkpoint (modele entier pickle) deja sauvegarde sous
pruning/checkpoints/ et le compare au meme modele baseline, sur les 50000
images completes, une fois avec --batch-size 128 puis une fois avec
--batch-size 1024 (deux JSON de sortie distincts), en reutilisant EXACTEMENT
la meme logique de mesure (resnet50_eval_lib.evaluate, NVML, warmup, repeats)
que les autres scripts, pour rester comparable.

Donnees (identiques aux autres scripts, lecture seule sur /raid) :
- train/ : uniquement pour deriver class_to_idx (ordre trie des synsets).
- val/ + LOC_val_solution.csv : pour l'evaluation.

Tout ce que ce script ecrit (JSON) va sous pruning/runs/, jamais sous /raid.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from cli_common import EXPECTED_TOP1_PCT, bootstrap_gpu_selection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, required=True,
                   help="Index physique du GPU a utiliser, tel qu'affiche par nvidia-smi")
    p.add_argument("--data-root", default="/raid/ILSVRC/Data/CLS-LOC")
    p.add_argument("--val-csv", default="/raid/LOC_val_solution.csv")
    p.add_argument("--weights", default="IMAGENET1K_V2", choices=sorted(EXPECTED_TOP1_PCT))
    p.add_argument("--checkpoint",
                   default=str(Path(__file__).resolve().parent / "checkpoints" /
                               "resnet50_pruned_IMAGENET1K_V2_0.25_20260915T122938.pt"),
                   help="Checkpoint pruned+finetune (modele entier pickle) a comparer a la baseline")
    p.add_argument("--batch-sizes", default="128,1024",
                   help="Liste de batch size d'inference a tester, dans l'ordre, separes par des virgules")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10,
                   help="Batches de warmup (tenseurs synthetiques), exclus de toute mesure")
    p.add_argument("--repeats", type=int, default=3,
                   help="Nombre de repetitions de la boucle mesuree (moyenne +/- ecart-type)")
    p.add_argument("--preload", action="store_true",
                   help="Precharge les tenseurs pretraites en RAM (~30 Go pour 50000 images) : "
                        "deconseille ici, laisse disponible pour usage avance.")
    p.add_argument("--out-dir", default=None,
                   help="Dossier de sortie des JSON (defaut : pruning/runs)")
    return p.parse_args()


_ARGS = parse_args()
bootstrap_gpu_selection(_ARGS.gpu)

# Import differe : torch ne doit voir qu'un seul GPU (fixe ci-dessus).
import torch  # noqa: E402
import torch_pruning as tp  # noqa: E402
from torchvision.models import ResNet50_Weights, resnet50  # noqa: E402

from resnet50_eval_lib import (  # noqa: E402
    build_class_to_idx,
    build_val_dataset,
    evaluate,
    init_gpu,
    preload_tensors,
    print_validity_check,
)


class _PreloadedDataset:
    """Vue en memoire (tenseurs deja decodes/transformes) d'un ImageNetValCSV.

    evaluate() appelle preload_tensors(dataset) a CHAQUE repeat et a chaque
    appel (baseline/pruned, par batch size) quand --preload est actif ; sur
    resnet50_eval_lib.ImageNetValCSV, __getitem__ relit et redecode le JPEG
    depuis le disque, en mono-thread (pas de DataLoader/workers). Sur 50000
    images, appele 12 fois (2 modeles x 2 batch sizes x repeats=3) dans ce
    script, ca reviendrait a redecoder 600000 JPEG en serie. En enveloppant
    les tenseurs deja precalcules UNE fois, chaque preload_tensors() ulterieur
    ne fait plus que de la reindexation/un stack de tenseurs deja en RAM."""

    def __init__(self, tensors: "torch.Tensor", labels: "torch.Tensor", transform) -> None:
        self.tensors = tensors
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return self.tensors.size(0)

    def __getitem__(self, idx: int):
        return self.tensors[idx], self.labels[idx]


def main() -> None:
    args = _ARGS
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    handle, gpu_name, device = init_gpu(args.gpu)

    weights = ResNet50_Weights[args.weights]
    eval_transform = weights.transforms()
    img_size = getattr(eval_transform, "crop_size", [224])[0]

    data_root = Path(args.data_root)
    class_to_idx = build_class_to_idx(data_root / "train")
    print(f"{len(class_to_idx)} classes indexees depuis {data_root / 'train'} (ordre trie des synsets)")

    val_dataset = build_val_dataset(data_root, args.val_csv, class_to_idx, eval_transform, n_images=0)

    if args.preload:
        print("Prechargement UNIQUE des 50000 images en tenseurs (evite de redecoder "
              "les JPEG a chaque repeat/appel a evaluate())...")
        tensors, labels = preload_tensors(val_dataset)
        eval_dataset = _PreloadedDataset(tensors, labels, eval_transform)
    else:
        eval_dataset = val_dataset

    baseline_model = resnet50(weights=weights).to(device).eval()

    print(f"Chargement du checkpoint pruned+finetune : {args.checkpoint}")
    # Architecture post-pruning != resnet50 vanilla : le checkpoint contient
    # le module entier pickle (voir resnet50_prune_finetune.py), pas un
    # state_dict. torch >= 2.6 : weights_only=True par defaut casserait ce
    # rechargement, d'ou weights_only=False (fichier que l'on a produit
    # nous-memes lors du run de pruning).
    pruned_model = torch.load(args.checkpoint, weights_only=False, map_location=device).eval()

    example_inputs = torch.randn(1, 3, img_size, img_size, device=device)
    base_macs, base_params = tp.utils.count_ops_and_params(baseline_model, example_inputs)
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(pruned_model, example_inputs)
    params_reduction_pct = (1 - pruned_params / base_params) * 100.0
    macs_reduction_pct = (1 - pruned_macs / base_macs) * 100.0
    print(f"Baseline : {base_params / 1e6:.2f} M params, {base_macs / 1e9:.2f} GMACs")
    print(f"Pruned+finetune : {pruned_params / 1e6:.2f} M params ({params_reduction_pct:.1f}% de reduction), "
          f"{pruned_macs / 1e9:.2f} GMACs ({macs_reduction_pct:.1f}% de reduction)")

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")

    for bs in batch_sizes:
        print(f"=== Evaluation sur les {len(eval_dataset)} images du val set, batch-size={bs} ===")
        eval_args = copy.copy(args)
        eval_args.batch_size = bs

        print(f"--- Baseline (modele pre-entraine intact) ---")
        baseline_results = evaluate(baseline_model, device, handle, gpu_name, eval_dataset, eval_args)
        print_validity_check(baseline_results["top1_acc"], args.weights)

        print(f"--- Pruned+finetune (checkpoint) ---")
        pruned_results = evaluate(pruned_model, device, handle, gpu_name, eval_dataset, eval_args)

        energy_reduction_pct = (1 - pruned_results["energy_j_mean"] / baseline_results["energy_j_mean"]) * 100.0
        report = {
            "gpu_index": args.gpu,
            "gpu_name": gpu_name,
            "weights": args.weights,
            "checkpoint_path": str(args.checkpoint),
            "eval_config": {
                "batch_size": bs,
                "num_workers": args.num_workers,
                "warmup_batches": args.warmup,
                "repeats": args.repeats,
                "preload": args.preload,
                "n_images": baseline_results["n_images"],
            },
            "baseline": {**baseline_results, "params": base_params, "macs": base_macs},
            "pruned_finetuned": {**pruned_results, "params": pruned_params, "macs": pruned_macs},
            "summary": {
                "params_reduction_pct": params_reduction_pct,
                "macs_reduction_pct": macs_reduction_pct,
                "energy_reduction_pct": energy_reduction_pct,
                "flops_vs_energy_reduction_gap_pct": macs_reduction_pct - energy_reduction_pct,
                "top1_delta_pct_points": (pruned_results["top1_acc"] - baseline_results["top1_acc"]) * 100.0,
                "top5_delta_pct_points": (pruned_results["top5_acc"] - baseline_results["top5_acc"]) * 100.0,
                "throughput_speedup_x":
                    pruned_results["throughput_img_s_mean"] / baseline_results["throughput_img_s_mean"],
            },
        }

        out_path = out_dir / f"resnet50_eval_full_{args.weights}_bs{bs}_{ts}.json"
        out_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report["summary"], indent=2))
        print(f"Rapport ecrit dans {out_path}")


if __name__ == "__main__":
    main()
