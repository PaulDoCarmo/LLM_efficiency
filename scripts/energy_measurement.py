"""Mesure de la consommation énergétique, de l'utilisation GPU et de la VRAM
pour un benchmark, via nvidia-smi. À appeler depuis n'importe quel script de
benchmark (un GPU par process — voir CLAUDE.md, pas de multi-GPU dans un
même run).

Usage :

    from energy_measurement import EnergyMeasurement

    with EnergyMeasurement(gpu_index=0, metadata={"model": "smollm2-360m", "quant": "bf16"}) as m:
        # chauffe déjà faite avant d'entrer ici : seule la boucle d'inférence
        # continue à mesurer va ici, sans I/O ni print à l'intérieur.
        run_inference_block(...)

    print(m.energy_j, m.mean_power_w, m.mean_utilization_pct, m.peak_vram_mib)
    # m.run_dir contient aussi power_trace.csv (brut), energy_timeseries.csv
    # (dérivé, une ligne par timestamp), summary.json et pip_freeze.txt

Protocole (voir CLAUDE.md, Yang et al. arXiv:2312.02741) :
- nvidia-smi échantillonné en tâche de fond à 100 ms.
- Refuse le GPU d'affichage (index 3) et tout GPU déjà occupé par un autre
  processus avant de démarrer (power.draw NVML est une mesure par carte,
  pas par process).
- Intègre la puissance sur les timestamps réels (jamais un pas constant).
- Rejette les 1,25 premières secondes de chaque bloc (250 ms de montée +
  1 s de fenêtre glissante NVML) et tout ce qui dépasse la fin du bloc.
- Le calcul d'énergie n'a lieu qu'à la sortie du bloc mesuré (__exit__),
  après l'arrêt du sampler — jamais pendant : aucune I/O ni print ne doit
  se produire à l'intérieur du `with`.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

NVIDIA_SMI_FIELDS = "timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used"
SAMPLE_PERIOD_MS = 100
SAMPLER_STARTUP_DELAY_S = 0.3
STARTUP_DISCARD_S = 1.25
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S.%f"

# Sur la DGX Station de l'équipe, l'index 3 est un GPU d'affichage
# ("NVIDIA DGX Display"), pas une carte de calcul A100 — voir CLAUDE.md.
FORBIDDEN_GPU_INDICES = {3}


def list_gpu_indices() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    return [int(line.strip()) for line in result.stdout.strip().splitlines() if line.strip()]


def gpu_processes(gpu_index: int, exclude_pid: int) -> list[str]:
    """Liste des process trouvés sur ce GPU, hors notre propre process (vide = libre)."""
    result = subprocess.run(
        ["nvidia-smi", "--id", str(gpu_index),
         "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    return [line for line in lines if not line.strip().startswith(f"{exclude_pid},")]


def check_gpu_ready(gpu_index: int) -> None:
    if gpu_index in FORBIDDEN_GPU_INDICES:
        sys.exit(f"GPU {gpu_index} interdit (carte d'affichage) — voir CLAUDE.md.")

    all_indices = list_gpu_indices()
    if gpu_index not in all_indices:
        sys.exit(f"GPU {gpu_index} inexistant sur cette machine — GPU détectés : {all_indices}.")

    # On a potentiellement déjà chargé un modèle sur ce GPU avant d'arriver ici
    # (chauffe), donc notre propre PID peut déjà apparaître dans la liste.
    procs = gpu_processes(gpu_index, exclude_pid=os.getpid())
    if procs:
        details = "\n".join(procs)
        sys.exit(f"GPU {gpu_index} déjà utilisé par un autre processus, mesure impossible :\n{details}")


def _load_trace(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "power.draw [W]": "power_w",
        # Le champ demandé est "clocks.sm" mais nvidia-smi le nomme
        # "clocks.current.sm" dans l'en-tête CSV (constaté sur la DGX).
        "clocks.current.sm [MHz]": "clock_sm_mhz",
        "temperature.gpu": "temperature_c",
        "utilization.gpu [%]": "utilization_pct",
        "memory.used [MiB]": "vram_used_mib",
    })
    expected = {"power_w", "clock_sm_mhz", "temperature_c", "utilization_pct", "vram_used_mib"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes après renommage dans {csv_path.name} : {missing}. "
            f"En-tête brut : {list(df.columns)}. Le nommage nvidia-smi a probablement changé."
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"].str.strip(), format=TIMESTAMP_FORMAT)
    return df.sort_values("timestamp").reset_index(drop=True)


class EnergyMeasurement:
    """Context manager : englobe uniquement le bloc d'inférence mesuré."""

    def __init__(self, gpu_index: int, output_dir: Path | str = "results", metadata: Optional[dict[str, Any]] = None):
        check_gpu_ready(gpu_index)
        self.gpu_index = gpu_index
        self.output_dir = Path(output_dir)
        self.metadata = metadata or {}

        self.run_dir: Optional[Path] = None
        self.block_start: Optional[dt.datetime] = None
        self.block_end: Optional[dt.datetime] = None
        self._process: Optional[subprocess.Popen] = None
        self._file_handle = None

        # Résultats, remplis à la sortie du bloc mesuré (voir __exit__).
        self.energy_j: Optional[float] = None
        self.mean_power_w: Optional[float] = None
        self.mean_utilization_pct: Optional[float] = None
        self.peak_vram_mib: Optional[float] = None
        self.mean_vram_mib: Optional[float] = None

    def __enter__(self) -> "EnergyMeasurement":
        self.run_dir = self.output_dir / dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "nvidia-smi",
            "--id", str(self.gpu_index),
            "--query-gpu", NVIDIA_SMI_FIELDS,
            "--format", "csv,nounits",
            "-lms", str(SAMPLE_PERIOD_MS),
        ]
        self._file_handle = open(self.run_dir / "power_trace.csv", "w")
        self._process = subprocess.Popen(cmd, stdout=self._file_handle, stderr=subprocess.DEVNULL)
        time.sleep(SAMPLER_STARTUP_DELAY_S)

        self.block_start = dt.datetime.now()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.block_end = dt.datetime.now()

        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._file_handle.close()

        self._analyze()

        summary = {
            "block_start": self.block_start.isoformat(),
            "block_end": self.block_end.isoformat(),
            "gpu_index": self.gpu_index,
            "energy_j": self.energy_j,
            "energy_wh": self.energy_j / 3600.0 if self.energy_j is not None else None,
            "mean_power_w": self.mean_power_w,
            "mean_utilization_pct": self.mean_utilization_pct,
            "peak_vram_mib": self.peak_vram_mib,
            "mean_vram_mib": self.mean_vram_mib,
            **self.metadata,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        pip_freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        (self.run_dir / "pip_freeze.txt").write_text(pip_freeze)

    def _analyze(self) -> None:
        """Intègre la puissance sur les timestamps réels. Appelé hors fenêtre de mesure."""
        df = _load_trace(self.run_dir / "power_trace.csv")
        window_start = pd.Timestamp(self.block_start) + pd.Timedelta(seconds=STARTUP_DISCARD_S)
        window_end = pd.Timestamp(self.block_end)
        windowed = df[(df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)]
        if len(windowed) < 2:
            raise ValueError("Pas assez d'échantillons dans la fenêtre de mesure après élagage du démarrage.")

        t = (windowed["timestamp"] - windowed["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
        p = windowed["power_w"].to_numpy()
        increments = np.concatenate([[0.0], np.diff(t) * (p[:-1] + p[1:]) / 2])

        timeseries = windowed[["timestamp", "power_w", "utilization_pct", "vram_used_mib"]].copy()
        timeseries["energy_j"] = np.cumsum(increments)
        timeseries.to_csv(self.run_dir / "energy_timeseries.csv", index=False)

        self.energy_j = float(timeseries["energy_j"].iloc[-1])
        self.mean_power_w = float(p.mean())
        self.mean_utilization_pct = float(windowed["utilization_pct"].mean())
        self.peak_vram_mib = float(windowed["vram_used_mib"].max())
        self.mean_vram_mib = float(windowed["vram_used_mib"].mean())

        clock_min, clock_max = windowed["clock_sm_mhz"].min(), windowed["clock_sm_mhz"].max()
        if clock_max > 0 and 100 * (1 - clock_min / clock_max) > 10:
            drop_pct = 100 * (1 - clock_min / clock_max)
            print(f"ATTENTION : décrochage d'horloge SM de {drop_pct:.1f}% pendant le bloc.")
