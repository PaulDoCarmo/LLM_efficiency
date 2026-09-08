"""Échantillonnage GPU en tâche de fond via nvidia-smi.

Protocole issu de Yang, Adámek, Armour, arXiv:2312.02741 : période 100 ms,
champs timestamp/power.draw/clocks.sm/temperature.gpu/utilization.gpu. On
laisse nvidia-smi gérer sa propre boucle plutôt que de sonder nous-mêmes,
pour ne pas ajouter de jitter côté Python.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

NVIDIA_SMI_FIELDS = "timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu"
DEFAULT_PERIOD_MS = 100


class PowerSampler:
    """Lance/arrête nvidia-smi en arrière-plan et écrit une trace CSV brute."""

    def __init__(self, output_csv: Path, gpu_index: int = 0, period_ms: int = DEFAULT_PERIOD_MS):
        self.output_csv = Path(output_csv)
        self.gpu_index = gpu_index
        self.period_ms = period_ms
        self._process: Optional[subprocess.Popen] = None
        self._file_handle = None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("PowerSampler déjà démarré.")
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "nvidia-smi",
            "--id", str(self.gpu_index),
            "--query-gpu", NVIDIA_SMI_FIELDS,
            "--format", "csv,nounits",
            "-lms", str(self.period_ms),
        ]
        self._file_handle = open(self.output_csv, "w")
        self._process = subprocess.Popen(cmd, stdout=self._file_handle, stderr=subprocess.DEVNULL)

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self) -> "PowerSampler":
        self.start()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.stop()
