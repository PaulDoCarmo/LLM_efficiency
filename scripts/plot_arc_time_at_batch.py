"""Histogramme du temps d'inférence ARC des variantes, à une batch size donnée.

    python3 scripts/plot_arc_time_at_batch.py --batch 2
"""
from pathlib import Path
import argparse
import json

from plot_arc_summary import INK, INK_2, MUTED, SURFACE, ROOT, load, plt, style
from plot_arc_batch_sweep import SERIES

DEFAULT_GLOBS = [
    "results/arc_Qwen2.5-1.5B-Instruct_*_batch*.json",
    "results/sweep_Qwen_Qwen2.5-1.5B-Instruct_arc_batch*.json",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--glob", nargs="+", default=DEFAULT_GLOBS)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    found, models = {}, set()
    for path in sorted({q for g in args.glob for q in ROOT.glob(g)}):
        data = load(path)
        if not data.get("arc") or data["arc_batch_size"] != args.batch:
            continue
        models.add(data["model"])
        for r in data["results"]:
            if "arc" in r:
                found[r["variant"]] = r
    assert found, f"aucun run ARC trouvé à batch {args.batch}"
    assert len(models) == 1, f"plusieurs modèles : {models}"
    model = models.pop()

    # Classement par temps : la comparaison est le sujet, l'ordre alphabétique
    # ne dirait rien. La couleur reste celle de la variante, jamais celle du
    # rang, pour rester cohérente avec les figures de balayage.
    rows = sorted(found.items(), key=lambda kv: kv[1]["arc_elapsed_s"])
    names = [v for v, _ in rows]
    times = [r["arc_elapsed_s"] for _, r in rows]
    colors = [SERIES[v][0] for v in names]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200, constrained_layout=True)
    ys = range(len(names))
    ax.barh(ys, times, height=0.62, color=colors)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(names)
    ax.set_xlabel("secondes")
    ax.set_xlim(0, max(times) * 1.16)
    ax.set_title(
        f"Temps d'inférence ARC-Challenge à batch {args.batch}", color=INK, pad=10
    )
    for y, t in zip(ys, times):
        ax.text(t + max(times) * 0.012, y, f"{t:.1f} s", va="center", ha="left",
                fontsize=9, color=INK_2)

    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, linestyle="-")

    ref = rows[0][1]["arc"]
    fig.suptitle(model, fontsize=12, color=INK_2)
    fig.text(0.5, -0.03,
             f"{ref['n']} questions, {ref['sequences_scored']} séquences, "
             f"{rows[0][1]['arc_forward_passes']} forwards par variante. "
             f"Même travail dans tous les cas.",
             ha="center", fontsize=8.5, color=MUTED)

    label = f"_{args.label}" if args.label else ""
    out = ROOT / "results" / f"arc_{model.replace('/', '_')}_temps_batch{args.batch}{label}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    print(f"écrit : {out}")


if __name__ == "__main__":
    main()
