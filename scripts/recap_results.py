"""Index consolidé de toutes les mesures : une ligne par (run, variante).

Les JSON par run restent la source de vérité — ils portent la configuration
complète et le chemin de la trace d'énergie. Ce script n'en produit qu'une
table de lecture, régénérable à tout moment, jamais un remplacement.

    python3 scripts/recap_results.py
"""
from pathlib import Path
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "recap_mesures.csv"

# Rattachement d'un fichier à sa campagne, dans l'ordre : le premier motif qui
# correspond gagne.
CAMPAGNES = [
    (r"^size_",                  "taille de modele"),
    (r"^sweepsort_",             "balayage batch ARC, trie par longueur"),
    (r"^arc_sweep_",             "balayage batch ARC"),
    (r"^ifeval_sweep_",          "balayage batch IFEval"),
    (r"^sweep_.*_arc_batch",     "balayage batch ARC (doublon, nomenclature ancienne)"),
    (r"^arc_Qwen.*_batch",       "balayage batch ARC prequantifie (partiel, ancien)"),
]

COLS = [
    "campagne", "benchmark", "modele", "variante", "batch", "score_pct",
    "acc_pct", "temps_s", "energie_j", "puissance_w", "util_pct", "vram_go",
    "unites_travail", "gpu", "fichier",
]


def campagne(name):
    for pattern, label in CAMPAGNES:
        if re.search(pattern, name):
            return label
    return "historique / ponctuel"


def rows_for(path):
    data = json.load(path.open(encoding="utf-8"))
    camp = campagne(path.name)
    gpu = ",".join(data.get("gpus") or [])
    for r in data["results"]:
        common = dict(
            campagne=camp, modele=data["model"], variante=r.get("variant"),
            energie_j=r.get("energy_j"), puissance_w=r.get("mean_power_w"),
            util_pct=r.get("mean_utilization_pct"), vram_go=r.get("vram_gb"),
            gpu=gpu, fichier=path.name,
        )
        if "arc" in r:
            yield {**common, "benchmark": "ARC-Challenge",
                   "batch": r.get("arc_batch_size_used"),
                   "score_pct": r["arc"]["acc_norm"] * 100,
                   "acc_pct": r["arc"]["acc"] * 100,
                   "temps_s": r.get("arc_elapsed_s"),
                   "unites_travail": r.get("arc_forward_passes")}
        if r.get("ifeval_score") is not None:
            yield {**common, "benchmark": "IFEval",
                   "batch": r.get("ifeval_batch_size_used"),
                   "score_pct": r["ifeval_score"] * 100, "acc_pct": None,
                   "temps_s": r.get("ifeval_elapsed_s"),
                   "unites_travail": r.get("ifeval_total_tokens")}


def main():
    rows = []
    for path in sorted((ROOT / "results").glob("*.json")):
        try:
            rows.extend(rows_for(path))
        except (KeyError, json.JSONDecodeError) as exc:
            print(f"ignoré : {path.name} ({exc})")

    def key(r):
        return (r["campagne"], r["benchmark"], r["modele"], str(r["variante"]),
                r["batch"] if r["batch"] is not None else -1)

    rows.sort(key=key)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in COLS})

    print(f"écrit : {OUT}  ({len(rows)} mesures)\n")
    counts = {}
    for r in rows:
        counts.setdefault((r["campagne"], r["benchmark"]), set()).add(
            (r["modele"], r["variante"], r["batch"])
        )
    for (camp, bench), points in sorted(counts.items()):
        print(f"  {camp:<52} {bench:<14} {len(points):>3} points")


if __name__ == "__main__":
    main()
