"""Figures ARC-Challenge : performance et énergie par variante, pour chaque
batch size, plus une comparaison entre les deux batches.

Attend deux runs complets du MÊME modèle, différant uniquement par le batch.
Le modèle, les batches et le nombre de questions sont lus dans les JSON : rien
n'est codé en dur, les figures sont nommées d'après le modèle pour que deux
modèles ne s'écrasent pas.

    python3 scripts/plot_arc_summary.py
    python3 scripts/plot_arc_summary.py --runs results/<run_a>.json results/<run_b>.json
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
# Paire par défaut : le modèle de base, batch 2 et batch 256.
DEFAULT_RUNS = [
    ROOT / "results" / "Qwen_Qwen2.5-1.5B_20260910-154320.json",
    ROOT / "results" / "Qwen_Qwen2.5-1.5B_20260910-153034.json",
]

# Palette de référence (mode clair), slots catégoriels 1 et 2. Validée :
# ΔE CVD 24.7, vision normale 33.6, contraste ≥ 3:1 sur la surface.
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
})


def load(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def style(ax):
    """Grille et axes en filets pleins, discrets, sous les marques."""
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)


def label_bars(ax, xs, ys, fmt, dy):
    for x, y in zip(xs, ys):
        ax.text(x, y + dy, fmt(y), ha="center", va="bottom", fontsize=8, color=INK_2)


def bar_panel(ax, variants, values, title, ylabel, fmt, color=SERIES_1):
    """Série unique sur des catégories nominales : une seule couleur."""
    xs = range(len(variants))
    ax.bar(xs, values, width=0.5, color=color)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(variants)
    ax.set_title(title, color=INK, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.22)
    label_bars(ax, xs, values, fmt, max(values) * 0.02)
    style(ax)


def figure_par_batch(batch, data, out_path):
    results = data["results"]
    variants = [r["variant"] for r in results]
    acc = [r["arc"]["acc"] for r in results]
    acc_err = [r["arc"]["acc_stderr"] for r in results]
    norm = [r["arc"]["acc_norm"] for r in results]
    norm_err = [r["arc"]["acc_norm_stderr"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=200, constrained_layout=True)

    # Deux séries (acc et acc_norm) : la couleur porte l'identité, légende requise.
    ax = axes[0][0]
    xs = range(len(variants))
    w = 0.30
    gap = 0.02  # écart de surface entre barres adjacentes
    left = [x - w / 2 - gap / 2 for x in xs]
    right = [x + w / 2 + gap / 2 for x in xs]
    ax.bar(left, acc, width=w, color=SERIES_1, label="acc",
           yerr=acc_err, ecolor=INK_2, capsize=3, error_kw={"linewidth": 1})
    ax.bar(right, norm, width=w, color=SERIES_2, label="acc_norm (principale)",
           yerr=norm_err, ecolor=INK_2, capsize=3, error_kw={"linewidth": 1})
    ax.set_xticks(list(xs))
    ax.set_xticklabels(variants)
    ax.set_title(f"Justesse ARC-Challenge ({results[0]['arc']['n']} questions)", color=INK, pad=8)
    ax.set_ylabel("Proportion de bonnes réponses")
    ax.set_ylim(0, 0.62)
    label_bars(ax, left, acc, lambda v: f"{v:.1%}", 0.022)
    label_bars(ax, right, norm, lambda v: f"{v:.1%}", 0.022)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    style(ax)

    bar_panel(axes[0][1], variants, [r["energy_j"] for r in results],
              "Énergie consommée sur le bloc mesuré", "joules", lambda v: f"{v:,.0f}".replace(",", " "))
    # Temps du scoring ARC seul, hors chargement du modèle et hors chauffe.
    bar_panel(axes[0][2], variants, [r["arc_elapsed_s"] for r in results],
              "Temps d'inférence ARC", "secondes", lambda v: f"{v:.0f} s")
    bar_panel(axes[1][0], variants, [r["mean_power_w"] for r in results],
              "Puissance moyenne", "watts", lambda v: f"{v:.0f}")
    bar_panel(axes[1][1], variants, [r["mean_utilization_pct"] for r in results],
              "Utilisation GPU moyenne", "%", lambda v: f"{v:.0f}")
    bar_panel(axes[1][2], variants, [r["vram_gb"] for r in results],
              "VRAM, pic alloué par torch", "Go", lambda v: f"{v:.1f}")

    fwd = results[0]["arc_forward_passes"]
    dur = f"{min(r['arc_elapsed_s'] for r in results):.0f}–{max(r['arc_elapsed_s'] for r in results):.0f} s"
    fig.suptitle(f"{data['model']} — ARC-Challenge, batch {batch}", fontsize=14, color=INK)
    fig.text(0.5, -0.015,
             f"{fwd} forwards par variante · bloc mesuré {dur} · barres d'erreur = erreur-type binomiale",
             ha="center", fontsize=8.5, color=MUTED)
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"écrit : {out_path}")


def figure_comparaison(runs, out_path):
    batches = sorted(runs)
    variants = [r["variant"] for r in runs[batches[0]]["results"]]
    colors = {batches[0]: SERIES_1, batches[1]: SERIES_2}
    by = {b: {r["variant"]: r for r in runs[b]["results"]} for b in batches}

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=200, constrained_layout=True)
    xs = list(range(len(variants)))

    # Valeurs très proches et incertitude à montrer : points + barres d'erreur,
    # axe resserré. Des barres depuis zéro écraseraient l'écart et l'erreur-type.
    ax = axes[0][0]
    for i, b in enumerate(batches):
        off = (i - 0.5) * 0.16
        vals = [by[b][v]["arc"]["acc_norm"] for v in variants]
        errs = [by[b][v]["arc"]["acc_norm_stderr"] for v in variants]
        ax.errorbar([x + off for x in xs], vals, yerr=errs, fmt="o", markersize=8,
                    color=colors[b], ecolor=colors[b], elinewidth=1.4, capsize=4,
                    linestyle="none", label=f"batch {b}", markeredgecolor=SURFACE,
                    markeredgewidth=1.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(variants)
    ax.set_xlim(-0.5, len(variants) - 0.5)
    ax.set_title("Justesse (acc_norm) — identique entre batches", color=INK, pad=8)
    ax.set_ylabel("Proportion de bonnes réponses")
    ax.legend(frameon=False, loc="lower left")
    style(ax)

    def grouped(ax, key, title, ylabel, fmt):
        w = 0.30
        gap = 0.02
        for i, b in enumerate(batches):
            off = (i - 0.5) * (w + gap)
            vals = [by[b][v][key] for v in variants]
            ax.bar([x + off for x in xs], vals, width=w, color=colors[b], label=f"batch {b}")
            label_bars(ax, [x + off for x in xs], vals, fmt, 0)
        ax.set_xticks(xs)
        ax.set_xticklabels(variants)
        ax.set_title(title, color=INK, pad=8)
        ax.set_ylabel(ylabel)
        top = max(by[b][v][key] for b in batches for v in variants)
        ax.set_ylim(0, top * 1.18)
        ax.legend(frameon=False, loc="upper left")
        style(ax)

    grouped(axes[0][1], "energy_j", "Énergie — le batch change tout",
            "joules", lambda v: f"{v:,.0f}".replace(",", " "))
    grouped(axes[0][2], "arc_elapsed_s", "Temps d'inférence ARC", "secondes", lambda v: f"{v:.0f}")
    grouped(axes[1][0], "mean_power_w", "Puissance moyenne", "watts", lambda v: f"{v:.0f}")
    grouped(axes[1][1], "mean_utilization_pct", "Utilisation GPU moyenne", "%", lambda v: f"{v:.0f}")
    grouped(axes[1][2], "vram_gb", "VRAM, pic alloué par torch", "Go", lambda v: f"{v:.1f}")

    ref = runs[batches[0]]["results"][0]["arc"]
    ratio = batches[1] // batches[0]
    fig.suptitle(
        f"{runs[batches[0]]['model']} — ARC-Challenge, batch {batches[0]} contre batch {batches[1]}",
        fontsize=14, color=INK,
    )
    fig.text(0.5, -0.015,
             f"Même travail utile dans les deux cas : {ref['n']} questions, {ref['sequences_scored']} séquences. "
             f"L'énergie par forward n'est pas comparable entre batches "
             f"(un forward à {batches[1]} fait {ratio}× plus de travail).",
             ha="center", fontsize=8.5, color=MUTED)
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"écrit : {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs=2, type=Path, default=DEFAULT_RUNS,
                    help="Les deux JSON de résultats à comparer (même modèle, batches différents).")
    args = ap.parse_args()

    runs = {}
    for path in args.runs:
        data = load(path)
        batch = data["arc_batch_size"]
        assert data.get("arc"), f"{path.name} n'est pas un run ARC"
        assert batch not in runs, f"les deux runs ont le même batch ({batch})"
        runs[batch] = data

    models = {d["model"] for d in runs.values()}
    assert len(models) == 1, f"les deux runs doivent porter sur le même modèle : {models}"
    # Préfixe tiré du modèle : sans lui, deux modèles écriraient les mêmes PNG.
    prefix = models.pop().replace("/", "_")

    for batch, data in sorted(runs.items()):
        figure_par_batch(batch, data, ROOT / "results" / f"arc_{prefix}_batch{batch}_summary.png")
    figure_comparaison(runs, ROOT / "results" / f"arc_{prefix}_comparison.png")


if __name__ == "__main__":
    main()
