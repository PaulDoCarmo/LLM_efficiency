#!/usr/bin/env bash
# Lance le benchmark avec le Python du venv sur les GPU 0,1,2,4.
set -euo pipefail

# Racine du projet = dossier de ce script
cd "$(dirname "$0")"

# Python du venv (adapte le chemin si ton venv est ailleurs)
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "Python du venv introuvable a $PY - cree/active le venv d'abord." >&2
    exit 1
fi

# Aligne la numerotation CUDA sur celle de nvidia-smi (sinon l'ordre "le plus
# rapide d'abord" de CUDA peut ne pas correspondre aux indices ci-dessous).
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# GPU visibles
export CUDA_VISIBLE_DEVICES=0,1,2,4

# Args passés à ce script transmis tels quels au benchmark
# (ex: ./run_benchmark.sh --max-tokens 4000)
"$PY" benchmark.py "$@"