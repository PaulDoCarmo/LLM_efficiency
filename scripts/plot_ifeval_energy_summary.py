from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = ROOT / "results" / "Qwen_Qwen2.5-1.5B_20260909-004734.json"
ENERGY_ROOT = ROOT / "results" / "energy" / "Qwen_Qwen2.5-1.5B"
OUT_PATH = ROOT / "results" / "ifeval_energy_summary.png"

with RESULTS_JSON.open("r", encoding="utf-8") as f:
    bench = json.load(f)

energy_by_variant = {}
for variant_dir in sorted(ENERGY_ROOT.iterdir()):
    if not variant_dir.is_dir():
        continue
    variant = variant_dir.name
    candidates = sorted(variant_dir.iterdir(), key=lambda p: p.name)
    if not candidates:
        continue
    latest = candidates[-1]
    summary_path = latest / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as sf:
            summary = json.load(sf)
        energy_by_variant[variant] = {
            "energy_wh": float(summary["energy_wh"]),
            "mean_power_w": float(summary["mean_power_w"]),
            "mean_utilization_pct": float(summary["mean_utilization_pct"]),
            "peak_vram_mib": float(summary["peak_vram_mib"]),
        }

entries = []
for item in bench["results"]:
    variant = item["variant"]
    score = item.get("ifeval_score")
    if score is None:
        score = item["ifeval"].get("prompt_level_strict_acc,none")
    energy = energy_by_variant.get(variant, {})
    entries.append(
        {
            "variant": variant,
            "ifeval": float(score),
            "energy_wh": energy.get("energy_wh", 0.0),
            "mean_power_w": energy.get("mean_power_w", 0.0),
            "mean_utilization_pct": energy.get("mean_utilization_pct", 0.0),
        }
    )

variants = [e["variant"] for e in entries]
ifeval = [e["ifeval"] for e in entries]
energy_wh = [e["energy_wh"] for e in entries]
power_w = [e["mean_power_w"] for e in entries]

fig, axes = plt.subplots(3, 1, figsize=(9, 11), dpi=200, constrained_layout=True)

colors = ["#2E86DE", "#4CAF50", "#FFB703", "#E85D75"]

axes[0].bar(variants, ifeval, color=colors)
axes[0].set_title("IFEval – prompt-level strict accuracy")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, max(ifeval) * 1.3 if ifeval else 1)
for i, v in enumerate(ifeval):
    axes[0].text(i, v + 0.01, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
axes[0].grid(axis="y", linestyle="--", alpha=0.35)

axes[1].bar(variants, energy_wh, color=colors)
axes[1].set_title("Energy consumption")
axes[1].set_ylabel("Wh")
axes[1].grid(axis="y", linestyle="--", alpha=0.35)
for i, v in enumerate(energy_wh):
    axes[1].text(i, v + 5, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

axes[2].bar(variants, power_w, color=colors)
axes[2].set_title("Mean power draw")
axes[2].set_ylabel("W")
axes[2].grid(axis="y", linestyle="--", alpha=0.35)
for i, v in enumerate(power_w):
    axes[2].text(i, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

for ax in axes:
    ax.set_axisbelow(True)

fig.suptitle("Qwen/Qwen2.5-1.5B – IFEval and energy metrics", fontsize=14, y=1.02)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved combined figure to: {OUT_PATH}")
