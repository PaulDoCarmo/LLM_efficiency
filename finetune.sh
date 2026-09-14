#!/usr/bin/env bash
# Lance le finetuning QLoRA avec le Python du venv, sur UN SEUL GPU.
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

# UN SEUL GPU, contrairement a exec.sh : le Trainer de transformers bascule en
# nn.DataParallel des qu'il en voit plusieurs, et les poids 4 bits de
# bitsandbytes ne survivent pas a la replication (illegal memory access).
# GPU 3 interdit sur cette machine. Surcharge : GPU=1 ./finetune.sh
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

if [ "$CUDA_VISIBLE_DEVICES" = "3" ]; then
    echo "GPU 3 interdit sur cette machine - choisis 0, 1, 2 ou 4." >&2
    exit 1
fi

echo "finetuning sur le GPU $CUDA_VISIBLE_DEVICES"

# Args passes a ce script transmis tels quels
# (ex: ./finetune.sh --model Qwen/Qwen2.5-1.5B-Instruct --output-dir ...)
"$PY" finetuning/finetune_lora.py "$@"
