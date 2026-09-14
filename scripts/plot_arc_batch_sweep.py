"""Courbes ARC-Challenge en fonction de la batch size, une par quantization.

Lit tous les JSON d'un balayage (un run complet 4 variantes par batch) et trace
temps d'inférence, énergie, puissance, utilisation GPU, VRAM et justesse.

    python3 scripts/plot_arc_batch_sweep.py
    python3 scripts/plot_arc_batch_sweep.py --glob 'results/sweep_*_arc_batch*.json'
"""
from pathlib import Path
import argparse
import json

# Réutilise la palette validée et le style de l'autre script plutôt que de les
# redéfinir : les deux jeux de figures doivent rester visuellement cohérents.
from plot_arc_summary import (
    INK, INK_2, MUTED, SURFACE, SERIES_1, SERIES_2, ROOT, load, plt, style,
)

# Slots catégoriels 1 à 4 de la palette de référence. Validés en mode clair :
# pire paire adjacente ΔE 9.1 en vision daltonienne, 22.9 en vision normale.
# L'aqua et le jaune passent sous 3:1 de contraste sur la surface, d'où les
# étiquettes en bout de courbe (règle de relief) et un marqueur par série.
SERIES = {
    "fp16": (SERIES_1, "o"),
    "bf16": (SERIES_2, "s"),
    "int8": ("#1baf7a", "^"),
    "4bit": ("#eda100", "D"),
}
DEFAULT_GLOB = "results/sweep_*_arc_batch*.json"


def collect(paths):
    """{variante: [(batch, résultat), ...]} trié par batch, + nom du modèle."""
    by_variant, models = {}, set()
    for path in paths:
        data = load(path)
        assert data.get("arc"), f"{path.name} n'est pas un run ARC"
        models.add(data["model"])
        for r in data["results"]:
            if "arc" in r:  # une variante en échec n'a pas de résultat ARC
                by_variant.setdefault(r["variant"], []).append((data["arc_batch_size"], r))
    for rows in by_variant.values():
        rows.sort(key=lambda t: t[0])
    assert len(models) == 1, f"le balayage doit porter sur un seul modèle : {models}"
    return by_variant, models.pop()


def curve_panel(ax, by_variant, value, title, ylabel, fmt, logy=False):
    """Une courbe par quantization, échelle log en abscisse (batches en
    puissances de 2). Étiquette en bout de courbe : l'identité ne repose
    jamais sur la seule couleur."""
    for variant, (color, marker) in SERIES.items():
        rows = by_variant.get(variant)
        if not rows:
            continue
        xs = [b for b, _ in rows]
        ys = [value(r) for _, r in rows]
        ax.plot(xs, ys, color=color, marker=marker, markersize=5, linewidth=2,
                label=variant, markeredgecolor=SURFACE, markeredgewidth=1)
        # Texte en encre, jamais en couleur de série : c'est le marqueur voisin
        # qui porte l'identité.
        ax.annotate(f" {variant} {fmt(ys[-1])}", (xs[-1], ys[-1]), fontsize=7.5,
                    color=INK_2, va="center", ha="left", annotation_clip=False)

    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels([2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_xlabel("batch size (échelle log)")
    if logy:
        ax.set_yscale("log")
        ylabel += " (échelle log)"
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, pad=8)
    # Marge à droite pour les étiquettes de fin de courbe.
    ax.set_xlim(1.7, 900)
    ax.legend(frameon=False, loc="upper right" if not logy else "upper right", fontsize=8)
    style(ax)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, linestyle="-")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default=DEFAULT_GLOB,
                    help=f"Motif des JSON du balayage (défaut: {DEFAULT_GLOB}).")
    args = ap.parse_args()

    paths = sorted(ROOT.glob(args.glob))
    assert paths, f"aucun fichier ne correspond à {args.glob}"
    by_variant, model = collect(paths)
    batches = sorted({b for rows in by_variant.values() for b, _ in rows})
    print(f"{len(paths)} runs, batches {batches}")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), dpi=200, constrained_layout=True)

    curve_panel(axes[0][0], by_variant, lambda r: r["arc_elapsed_s"],
                "Temps d'inférence ARC", "secondes", lambda v: f"{v:.0f}s", logy=True)
    curve_panel(axes[0][1], by_variant, lambda r: r["energy_j"],
                "Énergie consommée", "joules", lambda v: f"{v:.0f}J", logy=True)
    curve_panel(axes[0][2], by_variant, lambda r: r["mean_power_w"],
                "Puissance moyenne", "watts", lambda v: f"{v:.0f}W")
    curve_panel(axes[1][0], by_variant, lambda r: r["mean_utilization_pct"],
                "Utilisation GPU moyenne", "%", lambda v: f"{v:.0f}%")
    curve_panel(axes[1][1], by_variant, lambda r: r["vram_gb"],
                "VRAM, pic alloué par torch", "Go", lambda v: f"{v:.1f}Go")
    # Témoin : la justesse ne doit pas dépendre du batch. Si elle bouge, c'est
    # du bruit numérique fp16, pas un effet du batch.
    curve_panel(axes[1][2], by_variant, lambda r: r["arc"]["acc_norm"] * 100,
                "Justesse (acc_norm) — témoin, doit être plate", "%", lambda v: f"{v:.1f}%")

    ref = next(iter(by_variant.values()))[0][1]["arc"]
    fig.suptitle(f"{model} — ARC-Challenge en fonction de la batch size", fontsize=15, color=INK)
    fig.text(0.5, -0.015,
             f"{ref['n']} questions, {ref['sequences_scored']} séquences à chaque point : "
             f"seul le batch change, jamais le travail utile. "
             f"Un run complet des 4 quantizations par batch, mesure d'énergie sur GPU dédié.",
             ha="center", fontsize=9, color=MUTED)

    out = ROOT / "results" / f"arc_{model.replace('/', '_')}_batch_sweep.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    print(f"écrit : {out}")


if __name__ == "__main__":
    main()
