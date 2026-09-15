"""Effet de la taille du modèle, à quantization et batch fixés (bf16, batch 32).

Une courbe par benchmark. ARC et IFEval ayant des coûts d'un ordre de grandeur
différent, temps et énergie sont en ordonnée logarithmique : un axe linéaire
écraserait ARC, et deux axes verticaux inventeraient une corrélation.

    python3 scripts/plot_model_size.py
"""
from pathlib import Path
import argparse
import json

from matplotlib.lines import Line2D

from plot_arc_summary import (
    INK, INK_2, MUTED, SURFACE, SERIES_1, SERIES_2, ROOT, load, plt, style,
)

# Tailles nominales en milliards de paramètres : c'est l'abscisse.
SIZES = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0}
BENCHES = {
    "ARC-Challenge": (SERIES_1, "o", "arc"),
    "IFEval": (SERIES_2, "s", "ifeval"),
}


def collect():
    """{benchmark: {taille: résultat}}, en ignorant les runs absents."""
    out = {}
    for label, (_c, _m, kind) in BENCHES.items():
        for size in SIZES:
            path = ROOT / "results" / f"size_{size}_{kind}.json"
            if not path.exists():
                continue
            data = load(path)
            assert data["variants"] == ["bf16"], f"{path.name} n'est pas un run bf16 seul"
            out.setdefault(label, {})[size] = data["results"][0]
    assert out, "aucun run de la campagne taille trouvé"
    return out


def panel(ax, runs, value, title, ylabel, fmt, logy=False, log_scale=True):
    for label, (color, marker, _k) in BENCHES.items():
        rows = runs.get(label)
        if not rows:
            continue
        sizes = [s for s in SIZES if s in rows]
        xs = [SIZES[s] for s in sizes]
        ys = [value(rows[s]) for s in sizes]
        ax.plot(xs, ys, color=color, marker=marker, markersize=7, linewidth=2,
                label=label, markeredgecolor=SURFACE, markeredgewidth=1.5)
        for x, y in zip(xs, ys):
            ax.annotate(fmt(y), (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7.5, color=INK_2)

    if log_scale:
        ax.set_xscale("log")
        ax.set_xlabel("taille du modèle (échelle log)")
    else:
        ax.set_xlabel("taille du modèle (milliards de paramètres)")
    ax.set_xticks(list(SIZES.values()))
    ax.set_xticklabels(list(SIZES))
    if logy and log_scale:
        ax.set_yscale("log")
        ylabel += " (échelle log)"
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, pad=10)
    ax.set_xlim(0.4, 9.2) if log_scale else ax.set_xlim(0, 7.8)
    style(ax)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, linestyle="-")


def score(r):
    return (r["arc"]["acc_norm"] if "arc" in r else r["ifeval_score"]) * 100


def elapsed(r):
    return r["arc_elapsed_s"] if "arc" in r else r["ifeval_elapsed_s"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=["log", "linear"], default="log",
                    help="'log' (défaut) : les deux benchmarks tiennent sur un même "
                    "axe malgré deux ordres de grandeur d'écart. 'linear' : montre "
                    "les écarts absolus, mais tasse ARC contre zéro. Fichier distinct.")
    args = ap.parse_args()
    ls = args.scale == "log"
    runs = collect()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=200, constrained_layout=True)
    panel(axes[0][0], runs, score, "Justesse", "%", lambda v: f"{v:.1f}", log_scale=ls)
    panel(axes[0][1], runs, elapsed, "Temps d'inférence", "secondes",
          lambda v: f"{v:.0f}s", logy=True, log_scale=ls)
    panel(axes[0][2], runs, lambda r: r["energy_j"], "Énergie consommée", "joules",
          lambda v: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}", logy=True, log_scale=ls)
    panel(axes[1][0], runs, lambda r: r["mean_power_w"], "Puissance moyenne", "watts",
          lambda v: f"{v:.0f}", log_scale=ls)
    panel(axes[1][1], runs, lambda r: r["mean_utilization_pct"], "Utilisation GPU moyenne",
          "%", lambda v: f"{v:.0f}", log_scale=ls)
    panel(axes[1][2], runs, lambda r: r["vram_gb"], "VRAM, pic alloué par torch", "Go",
          lambda v: f"{v:.1f}", log_scale=ls)

    handles = [
        Line2D([], [], color=c, marker=m, markersize=7, linewidth=2, label=label)
        for label, (c, m, _k) in BENCHES.items() if label in runs
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.012),
               ncol=len(handles), frameon=False, fontsize=11)

    fig.suptitle("Qwen2.5-Instruct — effet de la taille du modèle (bf16, batch 32)",
                 fontsize=15, color=INK)
    fig.text(0.5, -0.055,
             "Une seule variable change : le nombre de paramètres. Même quantization, "
             "même batch, même GPU, un run mesuré par benchmark. "
             "ARC : 1172 questions scorées par log-vraisemblance. IFEval : 541 prompts générés.",
             ha="center", fontsize=9, color=MUTED)

    out = ROOT / "results" / f"taille_modele_bf16_batch32{'' if ls else '_lineaire'}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    print(f"écrit : {out}")


if __name__ == "__main__":
    main()
