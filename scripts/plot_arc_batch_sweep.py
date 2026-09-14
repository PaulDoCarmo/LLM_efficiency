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
from matplotlib.lines import Line2D

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


def _place_end_labels(ax, endpoints, fmt):
    """Étiquettes de fin de courbe, écartées verticalement quand deux valeurs
    sont trop proches. Sans ça fp16 et bf16, presque toujours à égalité, se
    superposent et deviennent illisibles.

    Le calcul se fait en fraction d'axe, donc APRÈS que l'échelle et les
    bornes soient fixées."""
    if not endpoints:
        return
    to_axes = ax.transAxes.inverted()
    rows = []
    for variant, x, y in endpoints:
        _, fy = to_axes.transform(ax.transData.transform((x, y)))
        rows.append([fy, variant, x, y])
    rows.sort()
    gap = 0.055  # hauteur minimale entre deux étiquettes, en fraction d'axe
    for i in range(1, len(rows)):
        rows[i][0] = max(rows[i][0], rows[i - 1][0] + gap)
    for fy, variant, x, y in rows:
        # Texte en encre, jamais en couleur de série : c'est le marqueur voisin
        # qui porte l'identité.
        ax.annotate(f" {variant} {fmt(y)}", xy=(x, fy), xycoords=("data", "axes fraction"),
                    fontsize=7.5, color=INK_2, va="center", ha="left", annotation_clip=False)


def ref_stderr(by_variant):
    """Erreur-type binomiale d'un point : identique partout, n ne change pas."""
    return next(iter(by_variant.values()))[0][1]["arc"]["acc_norm_stderr"]


def curve_panel(ax, by_variant, value, title, ylabel, fmt, logy=False, min_span=None,
                log_scale=True):
    """Une courbe par quantization, échelle log en abscisse (batches en
    puissances de 2). Étiquette en bout de courbe : l'identité ne repose
    jamais sur la seule couleur (l'aqua et le jaune passent sous 3:1 de
    contraste, la règle de relief impose un libellé visible)."""
    endpoints = []
    for variant, (color, marker) in SERIES.items():
        rows = by_variant.get(variant)
        if not rows:
            continue
        xs = [b for b, _ in rows]
        ys = [value(r) for _, r in rows]
        ax.plot(xs, ys, color=color, marker=marker, markersize=5, linewidth=2,
                label=variant, markeredgecolor=SURFACE, markeredgewidth=1)
        endpoints.append((variant, xs[-1], ys[-1]))

    if log_scale:
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
        ax.set_xticklabels([2, 4, 8, 16, 32, 64, 128, 256])
        ax.set_xlabel("batch size (échelle log)")
    else:
        # Graduations régulières : en linéaire, étiqueter 2, 4, 8, 16 les
        # ferait se chevaucher, elles tombent toutes dans le premier dixième
        # de l'axe.
        ax.set_xticks([0, 32, 64, 96, 128, 160, 192, 224, 256])
        ax.set_xlabel("batch size (échelle linéaire)")
    if logy and log_scale:
        ax.set_yscale("log")
        ylabel += " (échelle log)"
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, pad=8)
    # Marge à droite pour les étiquettes de fin de courbe.
    ax.set_xlim(1.7, 950) if log_scale else ax.set_xlim(-8, 400)
    if min_span:
        # Impose une amplitude minimale en ordonnée : sur un axe qui s'ajuste
        # aux données, quelques dixièmes de point de bruit remplissent le
        # panneau et donnent l'illusion d'un effet. Ici l'échelle est celle de
        # l'erreur-type, donc l'œil juge la variation à la bonne aune.
        lo, hi = ax.get_ylim()
        if hi - lo < min_span:
            mid = (hi + lo) / 2
            ax.set_ylim(mid - min_span / 2, mid + min_span / 2)
    style(ax)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, linestyle="-")
    _place_end_labels(ax, endpoints, fmt)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default=DEFAULT_GLOB,
                    help=f"Motif des JSON du balayage (défaut: {DEFAULT_GLOB}).")
    ap.add_argument("--scale", choices=["log", "linear"], default="log",
                    help="Échelle des axes. 'log' (défaut) linéarise les lois de "
                    "puissance ; 'linear' montre les écarts absolus mais tasse "
                    "les petits batches à gauche. Écrit un fichier distinct.")
    args = ap.parse_args()
    log_scale = args.scale == "log"

    paths = sorted(ROOT.glob(args.glob))
    assert paths, f"aucun fichier ne correspond à {args.glob}"
    by_variant, model = collect(paths)
    batches = sorted({b for rows in by_variant.values() for b, _ in rows})
    print(f"{len(paths)} runs, batches {batches}")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), dpi=200, constrained_layout=True)

    curve_panel(axes[0][0], by_variant, lambda r: r["arc_elapsed_s"],
                "Temps d'inférence ARC", "secondes", lambda v: f"{v:.0f}s", logy=True,
                log_scale=log_scale)
    curve_panel(axes[0][1], by_variant, lambda r: r["energy_j"],
                "Énergie consommée", "joules", lambda v: f"{v:.0f}J", logy=True,
                log_scale=log_scale)
    curve_panel(axes[0][2], by_variant, lambda r: r["mean_power_w"],
                "Puissance moyenne", "watts", lambda v: f"{v:.0f}W", log_scale=log_scale)
    curve_panel(axes[1][0], by_variant, lambda r: r["mean_utilization_pct"],
                "Utilisation GPU moyenne", "%", lambda v: f"{v:.0f}%", log_scale=log_scale)
    curve_panel(axes[1][1], by_variant, lambda r: r["vram_gb"],
                "VRAM, pic alloué par torch", "Go", lambda v: f"{v:.1f}Go", log_scale=log_scale)
    # Témoin : la justesse ne doit pas dépendre du batch. Si elle bouge, c'est
    # du bruit numérique fp16, pas un effet du batch.
    # Échelle imposée à ±2 erreurs-types : toute la variation doit tenir
    # dedans, sinon le batch aurait un vrai effet sur la mesure.
    stderr_pp = ref_stderr(by_variant) * 100
    curve_panel(axes[1][2], by_variant, lambda r: r["arc"]["acc_norm"] * 100,
                f"Justesse (acc_norm) — témoin, échelle ±2 erreurs-types ({stderr_pp:.2f} pt)",
                "%", lambda v: f"{v:.1f}%", min_span=4 * stderr_pp, log_scale=log_scale)

    handles = [
        Line2D([], [], color=color, marker=marker, markersize=5, linewidth=2, label=variant)
        for variant, (color, marker) in SERIES.items()
        if variant in by_variant
    ]
    # Sous les axes plutôt qu'au-dessus : posée dans la figure, une légende
    # unique recouvrirait un panneau (constrained_layout ne lui réserve pas de
    # place, et les positions "outside" demandent matplotlib >= 3.7).
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.012),
               ncol=len(handles), frameon=False, fontsize=10)

    ref = next(iter(by_variant.values()))[0][1]["arc"]
    fig.suptitle(f"{model} — ARC-Challenge en fonction de la batch size", fontsize=15, color=INK)
    fig.text(0.5, -0.055,
             f"{ref['n']} questions, {ref['sequences_scored']} séquences à chaque point : "
             f"seul le batch change, jamais le travail utile. "
             f"Un run complet des 4 quantizations par batch, mesure d'énergie sur GPU dédié.",
             ha="center", fontsize=9, color=MUTED)

    suffix = "" if log_scale else "_lineaire"
    out = ROOT / "results" / f"arc_{model.replace('/', '_')}_batch_sweep{suffix}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    print(f"écrit : {out}")


if __name__ == "__main__":
    main()
