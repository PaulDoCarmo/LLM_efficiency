from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "results" / "Qwen_Qwen2.5-1.5B_20260909-004734.json"
OUT_PATH = ROOT / "results" / "ifeval_summary.png"

with JSON_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

entries = []
for item in data["results"]:
    score = item.get("ifeval_score")
    if score is None:
        score = item["ifeval"].get("prompt_level_strict_acc,none")
    entries.append({
        "variant": item["variant"],
        "score": float(score),
    })

entries.sort(key=lambda x: x["score"], reverse=True)
labels = [e["variant"] for e in entries]
values = [e["score"] for e in entries]
colors = ["#2E86DE", "#4CAF50", "#FFB703", "#E85D75"]

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
ax.bar(labels, values, color=colors[: len(labels)])
ax.set_ylim(0, max(values) * 1.25 if values else 1)
ax.set_ylabel("Prompt-level strict accuracy")
ax.set_title("IFEval – Qwen/Qwen2.5-1.5B")
ax.grid(axis="y", linestyle="--", alpha=0.4)

for i, v in enumerate(values):
    ax.text(i, v + 0.01, f"{v:.1%}", ha="center", va="bottom", fontsize=10)

fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved figure to: {OUT_PATH}")
