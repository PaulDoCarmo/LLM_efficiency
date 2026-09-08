"""Analyse d'une trace de puissance GPU produite par measure_dummy_inference.py.

Ne relit que les fichiers bruts (power_trace.csv, summary.json). Aucune
mesure ici : intégration sur les timestamps réels (jamais un pas constant),
rejet des 1,25 premières secondes (250 ms de montée + 1 s de fenêtre
glissante NVML) et de tout ce qui dépasse la fin du bloc mesuré.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Constaté sur notre machine ; à vérifier sur la DGX car le format peut
# varier selon la version du driver nvidia-smi.
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S.%f"
STARTUP_DISCARD_S = 1.25


def load_trace(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "power_trace.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "power.draw [W]": "power_w",
        "clocks.sm [MHz]": "clock_sm_mhz",
        "temperature.gpu": "temperature_c",
        "utilization.gpu [%]": "utilization_pct",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"].str.strip(), format=TIMESTAMP_FORMAT)
    return df.sort_values("timestamp").reset_index(drop=True)


def integrate_energy(df: pd.DataFrame, block_start: pd.Timestamp, block_end: pd.Timestamp) -> dict:
    window_start = block_start + pd.Timedelta(seconds=STARTUP_DISCARD_S)
    windowed = df[(df["timestamp"] >= window_start) & (df["timestamp"] <= block_end)]
    if len(windowed) < 2:
        raise ValueError("Pas assez d'échantillons dans la fenêtre de mesure après élagage du démarrage.")

    t = (windowed["timestamp"] - windowed["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    p = windowed["power_w"].to_numpy()
    energy_j = float(np.trapz(p, t))

    return {
        "energy_j": energy_j,
        "energy_wh": energy_j / 3600.0,
        "duration_measured_s": float(t[-1] - t[0]),
        "mean_power_w": float(p.mean()),
        "n_samples": int(len(windowed)),
        "clock_sm_min_mhz": float(windowed["clock_sm_mhz"].min()),
        "clock_sm_max_mhz": float(windowed["clock_sm_mhz"].max()),
        "temperature_max_c": float(windowed["temperature_c"].max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Dossier produit par measure_dummy_inference.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = json.loads((args.run_dir / "summary.json").read_text())
    df = load_trace(args.run_dir)

    block_start = pd.Timestamp(summary["block_start"])
    block_end = pd.Timestamp(summary["block_end"])

    result = integrate_energy(df, block_start, block_end)

    if result["clock_sm_max_mhz"] > 0:
        clock_drop_pct = 100 * (1 - result["clock_sm_min_mhz"] / result["clock_sm_max_mhz"])
        if clock_drop_pct > 10:
            print(f"ATTENTION : décrochage d'horloge SM de {clock_drop_pct:.1f}% pendant le bloc.")

    print(json.dumps({**summary, **result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
