"""Test de bout en bout pour energy_measurement.py : modèle PyTorch factice
mesuré via EnergyMeasurement.

Sert à valider la couche de mesure d'énergie avant qu'elle soit utilisée par
les vrais scripts de benchmark. Un GPU par process — voir CLAUDE.md, pas de
multi-GPU dans un même run.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from energy_measurement import EnergyMeasurement

WARMUP_SECONDS = 5.0


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
    print(f"[1/4] GPU {args.gpu_index}, bloc mesuré de {args.duration:.0f}s, batch={args.batch_size}")

    if not torch.cuda.is_available():
        sys.exit("CUDA indisponible : ce script doit tourner sur la machine GPU cible.")

    device = torch.device(f"cuda:{args.gpu_index}")
    torch.cuda.set_device(device)
    print(f"[2/4] Device : {torch.cuda.get_device_name(device)}")

    model = build_dummy_model(args.hidden_size, args.n_layers).to(device).eval()
    dummy_input = torch.randn(args.batch_size, args.hidden_size, device=device)

    # Garde-fou : détecte un offload silencieux vers un autre device.
    devices = {p.device for p in model.parameters()}
    if devices != {device}:
        sys.exit(f"Offload détecté : paramètres sur {devices}, attendu uniquement {device}.")

    print(f"[3/4] Chauffe hors mesure ({WARMUP_SECONDS:.0f}s)...")
    run_inference_block(model, dummy_input, WARMUP_SECONDS)

    print(f"[4/4] Bloc mesuré en cours ({args.duration:.0f}s, silence requis jusqu'à la fin)...")
    metadata = {
        "duration_requested_s": args.duration,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    with EnergyMeasurement(args.gpu_index, args.output_dir, metadata=metadata) as measurement:
        n_passes = run_inference_block(model, dummy_input, args.duration)
        measurement.metadata["n_forward_passes"] = n_passes

    print(f"Résultats dans {measurement.run_dir}")
    print(f"  {n_passes} forward passes")
    print(f"  Énergie : {measurement.energy_j:.1f} J ({measurement.energy_j / 3600:.4f} Wh)")
    print(f"  Puissance moyenne : {measurement.mean_power_w:.1f} W")
    print(f"  Utilisation moyenne : {measurement.mean_utilization_pct:.1f} %")
    print(f"  VRAM utilisée : {measurement.mean_vram_mib:.0f} MiB (pic {measurement.peak_vram_mib:.0f} MiB)")


if __name__ == "__main__":
    main()
