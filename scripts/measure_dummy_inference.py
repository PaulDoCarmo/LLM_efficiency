"""Mesure : modèle PyTorch factice + échantillonnage de la puissance GPU.

Séparation stricte mesure/analyse (voir CLAUDE.md) : ce script n'écrit que
des fichiers bruts (trace de puissance, résumé JSON, pip freeze). Aucun
calcul d'énergie ou de score ici — c'est le rôle de analyze_power_trace.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from power_sampler import PowerSampler

WARMUP_SECONDS = 5.0
SAMPLER_STARTUP_DELAY_S = 0.3

# Sur la DGX Station de l'équipe, l'index 3 est un GPU d'affichage
# ("NVIDIA DGX Display"), pas une carte de calcul A100 — voir CLAUDE.md.
FORBIDDEN_GPU_INDICES = {3}


def build_dummy_model(hidden_size: int, n_layers: int) -> torch.nn.Module:
    layers = []
    for _ in range(n_layers):
        layers.append(torch.nn.Linear(hidden_size, hidden_size))
        layers.append(torch.nn.GELU())
    return torch.nn.Sequential(*layers)


def run_inference_block(model: torch.nn.Module, dummy_input: torch.Tensor, duration_s: float) -> int:
    """Boucle de forward passes continue, sans I/O ni print, pendant `duration_s`."""
    n_passes = 0
    end_time = time.perf_counter() + duration_s
    with torch.no_grad():
        while time.perf_counter() < end_time:
            model(dummy_input)
            n_passes += 1
        torch.cuda.synchronize()
    return n_passes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0, help="Durée du bloc mesuré en secondes (>=60 recommandé)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--n-layers", type=int, default=24)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA indisponible : ce script doit tourner sur la machine GPU cible.")

    if args.gpu_index in FORBIDDEN_GPU_INDICES:
        sys.exit(f"GPU {args.gpu_index} interdit (carte d'affichage, pas de calcul) — utiliser 0, 1, 2 ou 4.")

    device = torch.device(f"cuda:{args.gpu_index}")
    torch.cuda.set_device(device)

    run_dir = args.output_dir / dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_dummy_model(args.hidden_size, args.n_layers).to(device).eval()
    dummy_input = torch.randn(args.batch_size, args.hidden_size, device=device)

    # Garde-fou : détecte un offload silencieux vers un autre device.
    devices = {p.device for p in model.parameters()}
    if devices != {device}:
        sys.exit(f"Offload détecté : paramètres sur {devices}, attendu uniquement {device}.")

    run_inference_block(model, dummy_input, WARMUP_SECONDS)  # chauffe hors mesure

    sampler = PowerSampler(run_dir / "power_trace.csv", gpu_index=args.gpu_index)
    sampler.start()
    time.sleep(SAMPLER_STARTUP_DELAY_S)

    block_start = dt.datetime.now()
    n_passes = run_inference_block(model, dummy_input, args.duration)
    block_end = dt.datetime.now()

    sampler.stop()

    summary = {
        "block_start": block_start.isoformat(),
        "block_end": block_end.isoformat(),
        "duration_requested_s": args.duration,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        "n_forward_passes": n_passes,
        "gpu_index": args.gpu_index,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    pip_freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (run_dir / "pip_freeze.txt").write_text(pip_freeze)

    print(f"Mesure terminée : {n_passes} forward passes, résultats dans {run_dir}")


if __name__ == "__main__":
    main()
