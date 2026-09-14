"""Dataset et boucle de mesure (accuracy + energie NVML) partages entre
resnet50_energy_baseline.py et resnet50_prune_finetune.py.

Module "lourd" : importe torch/numpy/pynvml/PIL. Ne JAMAIS importer ce
module avant que l'appelant ait fixe CUDA_VISIBLE_DEVICES (voir
cli_common.bootstrap_gpu_selection) — torch doit voir le bon (et unique) GPU
des son premier import dans le process.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, Tuple

import numpy as np
import pynvml
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


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


def build_val_dataset(data_root: Path, val_csv: Path, class_to_idx: dict[str, int],
                       transform, n_images: int) -> ImageNetValCSV:
    dataset = ImageNetValCSV(Path(data_root) / "val", val_csv, class_to_idx, transform, n_images=n_images)
    if n_images > 0 and len(dataset) < n_images:
        print(
            f"AVERTISSEMENT : --n-images {n_images} demande plus d'images que le val set "
            f"n'en contient ({len(dataset)} disponibles). Utilisation des {len(dataset)} images disponibles."
        )
    print(f"Dataset de validation : {len(dataset)} images")
    return dataset


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


def init_gpu(gpu_index: int):
    """Init NVML, prend le handle sur l'index PHYSIQUE (NVML ignore
    CUDA_VISIBLE_DEVICES), affiche/avertit sur l'etat du GPU. Retourne
    (handle, gpu_name, device) ou device est toujours "cuda:0" cote torch
    (le seul GPU visible, fixe par bootstrap_gpu_selection)."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    check_gpu_idle(handle, gpu_index)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    device = torch.device("cuda:0")
    return handle, gpu_name, device


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
    """Une passe complete sur batch_iter_fn(), sans mesure d'energie/temps.
    Reutilisable seul (hors evaluate()) pour un suivi d'accuracy pas cher,
    ex. entre les epochs de finetuning."""
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


def evaluate(model: torch.nn.Module, device: torch.device, handle, gpu_name: str,
             dataset: ImageNetValCSV, args: argparse.Namespace) -> dict:
    """Warmup (hors mesure) + boucle mesuree (args.repeats fois), NVML pour
    l'energie. Accuracy calculee une seule fois, au repeat 0, en reutilisant
    les logits deja produits. Ne fait ni init NVML, ni construction modele/
    dataset, ni ecriture JSON, ni controle de validite (celui-ci ne concerne
    que le modele pre-entraine, pas un modele pruned/finetune) : ca reste a
    la charge de l'appelant.

    Appelee A L'IDENTIQUE pour le modele baseline et le modele pruned+
    finetune : seul `model` change entre les deux appels si `args` (batch
    size, num_workers, warmup, repeats, preload) est le meme."""
    img_size = getattr(dataset.transform, "crop_size", [224])[0]

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

    return {
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


def print_validity_check(top1_acc: float, weights_name: str) -> None:
    """N'a de sens que pour le modele pre-entraine intact (pas pour un
    modele pruned/finetune, dont l'accuracy attendue n'est pas connue a
    priori)."""
    from cli_common import EXPECTED_TOP1_PCT, VALIDITY_TOLERANCE_PCT

    expected = EXPECTED_TOP1_PCT[weights_name]
    top1_pct = top1_acc * 100.0
    if abs(top1_pct - expected) > VALIDITY_TOLERANCE_PCT:
        print(
            f"AVERTISSEMENT : top-1 = {top1_pct:.2f}% tres eloigne de l'attendu "
            f"~{expected}% pour {weights_name}. C'est le symptome typique d'un "
            "mauvais alignement labels/classes (class_to_idx) : verifier que "
            "l'index de classe utilise l'ordre TRIE de train/, pas celui de "
            "LOC_synset_mapping.txt."
        )
    else:
        print(f"Controle de validite OK : top-1 = {top1_pct:.2f}% proche de l'attendu ~{expected}%.")
