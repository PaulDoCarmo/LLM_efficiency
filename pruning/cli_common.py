"""Constantes et helpers CLI partages entre resnet50_energy_baseline.py et
resnet50_prune_finetune.py. Module STDLIB UNIQUEMENT (aucun import
torch/numpy/pynvml/PIL) : il doit pouvoir etre importe avant que le script
appelant ait fixe CUDA_VISIBLE_DEVICES et importe torch.
"""
from __future__ import annotations

import argparse
import os
import sys

# Precision attendue (torchvision) pour le controle de validite en fin de run.
EXPECTED_TOP1_PCT = {"IMAGENET1K_V2": 80.4, "IMAGENET1K_V1": 76.1}
VALIDITY_TOLERANCE_PCT = 2.0

# GPU d'affichage sur cette station, pas une carte de calcul A100 : interdit.
FORBIDDEN_GPU_INDICES = {3}


def bootstrap_gpu_selection(gpu_index: int) -> None:
    """Doit tourner AVANT tout import de torch : fige le GPU visible via
    CUDA_VISIBLE_DEVICES pour que torch n'ait acces qu'a ce seul GPU physique
    (device "cuda:0" cote torch)."""
    if gpu_index in FORBIDDEN_GPU_INDICES:
        sys.exit(f"GPU {gpu_index} interdit : GPU d'affichage sur cette station, pas une carte de calcul.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)


def add_shared_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flags communs a tout script qui evalue un ResNet50 sur le val set dans
    les memes conditions que la baseline. Point d'ancrage unique : ne pas
    dupliquer ces add_argument ailleurs, pour garantir que baseline et script
    de pruning restent comparables (memes noms/defauts)."""
    parser.add_argument("--gpu", type=int, required=True,
                         help="Index physique du GPU a utiliser, tel qu'affiche par nvidia-smi")
    parser.add_argument("--data-root", default="/raid/ILSVRC/Data/CLS-LOC")
    parser.add_argument("--val-csv", default="/raid/LOC_val_solution.csv")
    parser.add_argument("--weights", default="IMAGENET1K_V2", choices=sorted(EXPECTED_TOP1_PCT))
    parser.add_argument("--n-images", type=int, default=0,
                         help="0 = tout le val set (50000) ; sinon les N premieres images")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10,
                         help="Batches de warmup (tenseurs synthetiques), exclus de toute mesure")
    parser.add_argument("--repeats", type=int, default=3,
                         help="Nombre de repetitions de la boucle mesuree (moyenne +/- ecart-type)")
    parser.add_argument("--preload", action="store_true",
                         help="Precharge les tenseurs pretraites en RAM et itere sans DataLoader "
                              "(mesure 'pur modele'). Le val complet (50k images, float32) pese "
                              "~30 Go de RAM : reserver --preload aux sous-ensembles via --n-images.")
    return parser
