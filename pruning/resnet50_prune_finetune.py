#!/usr/bin/env python3
"""Pruning structure (canaux) d'un ResNet50 pre-entraine, finetuning court sur
train/ pour recuperer l'accuracy, puis re-evaluation sur le val set EXACTEMENT
dans les memes conditions que resnet50_energy_baseline.py (meme evaluate(),
memes --batch-size/--num-workers/--warmup/--repeats/--preload/--gpu), afin de
produire une comparaison rigoureuse accuracy/energie baseline vs pruned.

Objectif d'etude : montrer une vraie baisse de consommation energetique a
l'inference pour une perte d'accuracy contenue, et mettre en evidence l'ecart
entre reduction de FLOPs (theorique) et reduction d'energie (mesuree).

Donnees (identiques a la baseline, lecture seule sur /raid) :
- train/ (ImageFolder, 1000 classes n########) : UNIQUEMENT pour le
  finetuning. INTERDICTION ABSOLUE de toucher au val/ pour l'entrainement.
- val/ + LOC_val_solution.csv : UNIQUEMENT pour l'evaluation (via le meme
  Dataset que la baseline).

Tout ce que ce script ecrit (checkpoint, JSON) va sous /home/fmr2026, jamais
sous /raid.
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
    p.add_argument("--pruning-ratio", type=float, default=0.5,
                   help="Taux de pruning cible (one-shot, global_pruning=True)")
    p.add_argument("--epochs", type=int, default=5, help="Epochs de finetuning post-pruning")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size-train", type=int, default=256)
    p.add_argument("--out", default=None,
                   help="Chemin du JSON comparatif (defaut : pruning/runs/<horodatage>.json)")
    return p.parse_args()


_ARGS = parse_args()
bootstrap_gpu_selection(_ARGS.gpu)

# Import differe : torch ne doit voir qu'un seul GPU (fixe ci-dessus). On
# n'importe JAMAIS resnet50_energy_baseline ici (il parse sys.argv a l'import
# et bootstraperait un second GPU) : uniquement cli_common (stdlib) et
# resnet50_eval_lib (le module lourd partage).
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch_pruning as tp  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402
from torchvision.models import ResNet50_Weights, resnet50  # noqa: E402

from resnet50_eval_lib import (  # noqa: E402
    build_class_to_idx,
    build_val_dataset,
    evaluate,
    init_gpu,
    make_batch_iter_fn,
    print_validity_check,
    run_measured_epoch,
)


def finetune(model, train_loader, val_dataset, device, args) -> None:
    """SGD + cosine + AMP. L'AMP (autocast/GradScaler) est scope strictement
    a cette boucle : resnet50_eval_lib ne contient aucun autocast, donc
    evaluate()/run_measured_epoch() (utilises ici pour le suivi par epoque,
    et plus tard pour la re-evaluation finale) tournent toujours en fp32/TF32
    pur, comme la baseline."""
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            n_seen += inputs.size(0)
        scheduler.step()

        model.eval()
        c1, c5, n = run_measured_epoch(model, device, make_batch_iter_fn(val_dataset, args), True)
        print(f"[epoch {epoch + 1}/{args.epochs}] loss={running_loss / n_seen:.4f} "
              f"val_top1={c1 / n:.4f} val_top5={c5 / n:.4f} (suivi leger, sans energie/temps)")


def main() -> None:
    args = _ARGS

    handle, gpu_name, device = init_gpu(args.gpu)

    weights = ResNet50_Weights[args.weights]
    eval_transform = weights.transforms()
    img_size = getattr(eval_transform, "crop_size", [224])[0]

    data_root = Path(args.data_root)
    class_to_idx = build_class_to_idx(data_root / "train")
    print(f"{len(class_to_idx)} classes indexees depuis {data_root / 'train'} (ordre trie des synsets)")

    val_dataset = build_val_dataset(data_root, args.val_csv, class_to_idx, eval_transform, args.n_images)

    model = resnet50(weights=weights).to(device).eval()

    print("=== Evaluation baseline (modele pre-entraine intact) ===")
    baseline_results = evaluate(model, device, handle, gpu_name, val_dataset, args)
    print_validity_check(baseline_results["top1_acc"], args.weights)

    example_inputs = torch.randn(1, 3, img_size, img_size, device=device)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Avant pruning : {base_params / 1e6:.2f} M params, {base_macs / 1e9:.2f} GMACs")

    print(f"=== Pruning structure (ratio={args.pruning_ratio}, global_pruning=True) ===")
    importance = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=args.pruning_ratio,
        global_pruning=True,
        ignored_layers=[model.fc],
    )
    pruner.step()

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    params_reduction_pct = (1 - pruned_params / base_params) * 100.0
    macs_reduction_pct = (1 - pruned_macs / base_macs) * 100.0
    print(f"Apres pruning : {pruned_params / 1e6:.2f} M params ({params_reduction_pct:.1f}% de reduction), "
          f"{pruned_macs / 1e9:.2f} GMACs ({macs_reduction_pct:.1f}% de reduction)")

    # Sanity check : confirme que fc s'est bien adapte a la nouvelle largeur
    # de features (ignored_layers protege la sortie 1000-way, pas l'entree).
    model.eval()
    with torch.inference_mode():
        test_out = model(example_inputs)
    assert test_out.shape[-1] == len(class_to_idx), (
        f"Sortie du modele pruned = {test_out.shape[-1]} classes, attendu {len(class_to_idx)}."
    )

    print("=== Evaluation du modele pruned, PAS ENCORE finetune ===")
    pruned_not_finetuned_results = evaluate(model, device, handle, gpu_name, val_dataset, args)
    print(f"Pruned (sans finetuning) : top1={pruned_not_finetuned_results['top1_acc']:.4f} "
          f"top5={pruned_not_finetuned_results['top5_acc']:.4f}")

    print(f"=== Finetuning ({args.epochs} epochs, lr={args.lr}, batch_size_train={args.batch_size_train}) ===")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=eval_transform.mean, std=eval_transform.std),
    ])
    train_dataset = ImageFolder(data_root / "train", transform=train_transform)
    assert train_dataset.class_to_idx == class_to_idx, (
        "class_to_idx d'ImageFolder(train/) ne correspond pas a l'index construit pour le val set : "
        "le finetuning entrainerait le modele sur un mauvais mapping classe->index."
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)

    finetune(model, train_loader, val_dataset, device, args)

    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    ckpt_path = ckpt_dir / f"resnet50_pruned_{args.weights}_{args.pruning_ratio}_{ts}.pt"

    model.eval()
    # Architecture post-pruning != resnet50 vanilla (formes de canaux
    # differentes) : state_dict() + resnet50().load_state_dict() echouerait.
    # On sauvegarde l'objet modele entier.
    torch.save(model, ckpt_path)
    print(f"Checkpoint (modele entier) sauvegarde dans {ckpt_path}")

    # torch >= 2.6 : torch.load a weights_only=True par defaut, ce qui casse
    # le rechargement d'un nn.Module picklé entier. weights_only=False
    # necessaire ici (fichier que l'on vient d'ecrire nous-memes).
    reloaded = torch.load(ckpt_path, weights_only=False, map_location=device).eval()
    assert sum(p.numel() for p in reloaded.parameters()) == sum(p.numel() for p in model.parameters()), (
        "Le modele recharge n'a pas le meme nombre de parametres que le modele sauvegarde."
    )

    print("=== Evaluation du modele pruned+finetune (rechargee depuis le checkpoint) ===")
    pruned_results = evaluate(reloaded, device, handle, gpu_name, val_dataset, args)

    energy_reduction_pct = (1 - pruned_results["energy_j_mean"] / baseline_results["energy_j_mean"]) * 100.0
    report = {
        "gpu_index": args.gpu,
        "gpu_name": gpu_name,
        "weights": args.weights,
        "pruning_ratio": args.pruning_ratio,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size_train": args.batch_size_train,
        "eval_config": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "warmup_batches": args.warmup,
            "repeats": args.repeats,
            "preload": args.preload,
            "n_images": baseline_results["n_images"],
        },
        "baseline": {**baseline_results, "params": base_params, "macs": base_macs},
        "pruned_not_finetuned": {**pruned_not_finetuned_results, "params": pruned_params, "macs": pruned_macs},
        "pruned_finetuned": {**pruned_results, "params": pruned_params, "macs": pruned_macs},
        "summary": {
            "params_reduction_pct": params_reduction_pct,
            "macs_reduction_pct": macs_reduction_pct,
            "energy_reduction_pct": energy_reduction_pct,
            "flops_vs_energy_reduction_gap_pct": macs_reduction_pct - energy_reduction_pct,
            "top1_delta_pct_points": (pruned_results["top1_acc"] - baseline_results["top1_acc"]) * 100.0,
            "top5_delta_pct_points": (pruned_results["top5_acc"] - baseline_results["top5_acc"]) * 100.0,
            "top1_drop_from_pruning_pct_points":
                (pruned_not_finetuned_results["top1_acc"] - baseline_results["top1_acc"]) * 100.0,
            "top1_recovered_by_finetuning_pct_points":
                (pruned_results["top1_acc"] - pruned_not_finetuned_results["top1_acc"]) * 100.0,
            "throughput_speedup_x": pruned_results["throughput_img_s_mean"] / baseline_results["throughput_img_s_mean"],
        },
        "checkpoint_path": str(ckpt_path),
    }

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path(__file__).resolve().parent / "runs"
        out_path = out_dir / f"resnet50_prune_{args.weights}_{args.pruning_ratio}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report["summary"], indent=2))
    print(f"Rapport comparatif ecrit dans {out_path}")


if __name__ == "__main__":
    main()
