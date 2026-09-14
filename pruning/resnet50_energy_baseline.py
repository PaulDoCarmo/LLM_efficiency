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
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, Tuple

# Precision attendue (torchvision) pour le controle de validite en fin de run.
EXPECTED_TOP1_PCT = {"IMAGENET1K_V2": 80.4, "IMAGENET1K_V1": 76.1}
VALIDITY_TOLERANCE_PCT = 2.0

# GPU d'affichage sur cette station, pas une carte de calcul A100 : interdit.
FORBIDDEN_GPU_INDICES = {3}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, required=True,
                   help="Index physique du GPU a utiliser, tel qu'affiche par nvidia-smi")
    p.add_argument("--data-root", default="/raid/ILSVRC/Data/CLS-LOC")
    p.add_argument("--val-csv", default="/raid/LOC_val_solution.csv")
    p.add_argument("--weights", default="IMAGENET1K_V2", choices=sorted(EXPECTED_TOP1_PCT))
    p.add_argument("--n-images", type=int, default=0,
                   help="0 = tout le val set (50000) ; sinon les N premieres images")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10,
                   help="Batches de warmup (tenseurs synthetiques), exclus de toute mesure")
    p.add_argument("--repeats", type=int, default=3,
                   help="Nombre de repetitions de la boucle mesuree (moyenne +/- ecart-type)")
    p.add_argument("--preload", action="store_true",
                   help="Precharge les tenseurs pretraites en RAM et itere sans DataLoader "
                        "(mesure 'pur modele'). Le val complet (50k images, float32) pese "
                        "~30 Go de RAM : reserver --preload aux sous-ensembles via --n-images.")
    p.add_argument("--out", default=None,
                   help="Chemin du JSON de sortie (defaut : pruning/runs/<horodatage>.json, "
                        "sous /home/fmr2026). Volontairement PAS pruning/results/ : ce nom est "
                        "couvert par le .gitignore du repo (convention des autres benchmarks) ; "
                        "pruning/runs/ reste versionne pour pouvoir push/pull les resultats "
                        "depuis le cluster.")
    return p.parse_args()


def _bootstrap_gpu_selection(gpu_index: int) -> None:
    """Doit tourner AVANT tout import de torch : fige le GPU visible via
    CUDA_VISIBLE_DEVICES pour que torch n'ait acces qu'a ce seul GPU physique
    (device "cuda:0" cote torch)."""
    if gpu_index in FORBIDDEN_GPU_INDICES:
        sys.exit(f"GPU {gpu_index} interdit : GPU d'affichage sur cette station, pas une carte de calcul.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)


_ARGS = parse_args()
_bootstrap_gpu_selection(_ARGS.gpu)

# Import differe : torch ne doit voir qu'un seul GPU (fixe ci-dessus). NVML,
# lui, enumere toujours TOUS les GPU physiques quel que soit CUDA_VISIBLE_DEVICES
# (voir check_gpu_idle / main : le handle NVML est pris sur l'index physique
# --gpu, pas sur 0).
import numpy as np  # noqa: E402
import pynvml  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision.models import ResNet50_Weights, resnet50  # noqa: E402


class ImageNetValCSV(Dataset):
    """Lit les labels du val set depuis LOC_val_solution.csv, sans jamais
    toucher a l'arborescence de /raid (val/ reste a plat)."""

    def __init__(self, val_dir: Path, val_csv: Path, class_to_idx: dict[str, int],
                 transform, n_images: int = 0):
        self.val_dir = Path(val_dir)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        with open(val_csv, newline="") as f:
            for row in csv.DictReader(f):
                synset = row["PredictionString"].split()[0]
                self.samples.append((row["ImageId"], class_to_idx[synset]))
        self.samples.sort(key=lambda s: s[0])
        if n_images > 0:
            self.samples = self.samples[:n_images]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_id, label = self.samples[idx]
        img = Image.open(self.val_dir / f"{image_id}.JPEG").convert("RGB")
        return self.transform(img), label


def check_gpu_idle(handle, gpu_index: int) -> None:
    """Affiche l'etat du GPU choisi et avertit (sans bloquer) si une autre
    charge tourne dessus : ca fausserait la mesure d'energie NVML, qui est
    par carte et non par process."""
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

    procs = []
    for getter in (pynvml.nvmlDeviceGetComputeRunningProcesses, pynvml.nvmlDeviceGetGraphicsRunningProcesses):
        try:
            procs.extend(getter(handle))
        except pynvml.NVMLError:
            pass

    print(f"GPU {gpu_index} : {gpu_name}")
    print(f"  Utilisation : {util.gpu}% compute, memoire : {mem.used / 2**20:.0f} / {mem.total / 2**20:.0f} MiB")
    if procs:
        print("  Process en cours sur ce GPU :")
        for proc in procs:
            used_mib = (proc.usedGpuMemory or 0) / 2**20
            print(f"    pid={proc.pid} memoire={used_mib:.0f} MiB")
    else:
        print("  Aucun process detecte.")

    if util.gpu > 5 or procs:
        print(
            f"AVERTISSEMENT : le GPU {gpu_index} n'est pas quasi idle "
            "(utilisation compute ou process detectes). Une charge concurrente "
            "sur cette carte partagee peut fausser la mesure d'energie de ce run."
        )


def build_class_to_idx(train_dir: Path) -> dict[str, int]:
    """Index de classe = ordre TRIE des synsets, pour correspondre a l'index
    des poids pre-entraines torchvision. NE PAS utiliser l'ordre de
    LOC_synset_mapping.txt (ordre different, accuracy effondree sans erreur)."""
    return {synset: i for i, synset in enumerate(sorted(os.listdir(train_dir)))}


def preload_tensors(dataset: ImageNetValCSV) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def make_batch_iter_fn(dataset: ImageNetValCSV, args: argparse.Namespace) -> Callable[[], Iterable]:
    if args.preload:
        print("Prechargement des tenseurs pretraites en RAM (--preload)...")
        tensors, labels = preload_tensors(dataset)
        n = tensors.size(0)

        def iter_batches() -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
            for start in range(0, n, args.batch_size):
                end = start + args.batch_size
                yield tensors[start:end], labels[start:end]

        return iter_batches

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    def iter_loader() -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        return iter(loader)

    return iter_loader


def warmup_model(model: torch.nn.Module, device: torch.device, n_batches: int,
                  batch_size: int, img_size: int) -> None:
    if n_batches <= 0:
        return
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)
    with torch.inference_mode():
        for _ in range(n_batches):
            model(dummy)
    torch.cuda.synchronize()


def run_measured_epoch(model: torch.nn.Module, device: torch.device,
                        batch_iter_fn: Callable[[], Iterable], compute_accuracy: bool):
    correct1 = correct5 = n = 0
    with torch.inference_mode():
        for inputs, labels in batch_iter_fn():
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            if compute_accuracy:
                labels_dev = labels.to(device, non_blocking=True)
                top5 = outputs.topk(5, dim=1).indices
                correct1 += (top5[:, 0] == labels_dev).sum().item()
                correct5 += (top5 == labels_dev.unsqueeze(1)).any(dim=1).sum().item()
                n += labels.size(0)
            else:
                n += inputs.size(0)
    return correct1, correct5, n


def measured_repeat(model: torch.nn.Module, device: torch.device, handle,
                     batch_iter_fn: Callable[[], Iterable], compute_accuracy: bool):
    torch.cuda.synchronize()
    e0_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    t0 = time.perf_counter()

    correct1, correct5, n = run_measured_epoch(model, device, batch_iter_fn, compute_accuracy)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    e1_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

    energy_j = (e1_mj - e0_mj) / 1000.0
    elapsed_s = t1 - t0
    return energy_j, elapsed_s, n, correct1, correct5


def main() -> None:
    args = _ARGS

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)  # index PHYSIQUE (NVML ignore CUDA_VISIBLE_DEVICES)
    check_gpu_idle(handle, args.gpu)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()

    device = torch.device("cuda:0")
    weights = ResNet50_Weights[args.weights]
    transform = weights.transforms()
    img_size = getattr(transform, "crop_size", [224])[0]

    data_root = Path(args.data_root)
    class_to_idx = build_class_to_idx(data_root / "train")
    print(f"{len(class_to_idx)} classes indexees depuis {data_root / 'train'} (ordre trie des synsets)")

    dataset = ImageNetValCSV(data_root / "val", args.val_csv, class_to_idx, transform, n_images=args.n_images)
    print(f"Dataset de validation : {len(dataset)} images")

    model = resnet50(weights=weights).to(device).eval()

    print(f"Warmup : {args.warmup} batches synthetiques (hors mesure)...")
    warmup_model(model, device, args.warmup, args.batch_size, img_size)

    energies_j, times_s, throughputs = [], [], []
    top1_acc = top5_acc = None
    n_images_measured = 0
    top1_correct = 0

    for repeat_idx in range(args.repeats):
        compute_acc = repeat_idx == 0
        batch_iter_fn = make_batch_iter_fn(dataset, args)
        energy_j, elapsed_s, n, c1, c5 = measured_repeat(model, device, handle, batch_iter_fn, compute_acc)
        energies_j.append(energy_j)
        times_s.append(elapsed_s)
        throughputs.append(n / elapsed_s)
        print(f"[repeat {repeat_idx + 1}/{args.repeats}] "
              f"energie={energy_j:.1f} J, temps={elapsed_s:.1f} s, debit={n / elapsed_s:.1f} img/s")
        if compute_acc:
            n_images_measured = n
            top1_correct = c1
            top1_acc = c1 / n
            top5_acc = c5 / n

    def mean(vals: list[float]) -> float:
        return float(np.mean(vals))

    def std(vals: list[float]) -> float:
        return float(np.std(vals))

    results = {
        "gpu_index": args.gpu,
        "gpu_name": gpu_name,
        "weights": args.weights,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "preload": args.preload,
        "warmup_batches": args.warmup,
        "repeats": args.repeats,
        "n_images": n_images_measured,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "energy_j_mean": mean(energies_j),
        "energy_j_std": std(energies_j),
        "time_s_mean": mean(times_s),
        "time_s_std": std(times_s),
        "throughput_img_s_mean": mean(throughputs),
        "throughput_img_s_std": std(throughputs),
        "latency_ms_per_image_mean": mean(times_s) / n_images_measured * 1000.0,
        "energy_j_per_image": mean(energies_j) / n_images_measured,
        "energy_j_per_correct_image_top1": (mean(energies_j) / top1_correct) if top1_correct else None,
    }

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

    expected = EXPECTED_TOP1_PCT[args.weights]
    top1_pct = top1_acc * 100.0
    if abs(top1_pct - expected) > VALIDITY_TOLERANCE_PCT:
        print(
            f"AVERTISSEMENT : top-1 = {top1_pct:.2f}% tres eloigne de l'attendu "
            f"~{expected}% pour {args.weights}. C'est le symptome typique d'un "
            "mauvais alignement labels/classes (class_to_idx) : verifier que "
            "l'index de classe utilise l'ordre TRIE de train/, pas celui de "
            "LOC_synset_mapping.txt."
        )
    else:
        print(f"Controle de validite OK : top-1 = {top1_pct:.2f}% proche de l'attendu ~{expected}%.")


if __name__ == "__main__":
    main()
