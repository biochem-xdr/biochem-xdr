import json
import glob
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as mp
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from pathlib import Path


TIER_COLORS = {
    "T1": "#4CAF50",
    "T2": "#2196F3",
    "T3": "#FF9800",
    "T4": "#F44336",
}
CAT_COLORS = {
    "frontier":   "#1565C0",
    "general":    "#2E7D32",
    "biomedical": "#6A1B9A",
}
COLORS = {
    "T1":        "#4CAF50",
    "T2":        "#2196F3",
    "T3":        "#FF9800",
    "T4":        "#F44336",
    "mcq":       "#1565C0",
    "openended": "#90CAF9",
    "random":    "#9E9E9E",
}

TIER_ORDER  = ["T1", "T2", "T3", "T4"]
TIER_LABELS = {"T1": "T1 (Easy)", "T2": "T2", "T3": "T3", "T4": "T4 (Hard)"}

FRONTIER_MODELS   = ["gpt-4o", "claude-sonnet-4.6", "gemini-2.5-flash",
                     "deepseek-v3", "qwen3-235b"]
GENERAL_MODELS    = ["llama-3.3-70b", "llama-3-8b-instruct", "mistral-7b-instruct",
                     "gemma-3-27b", "qwen3-32b"]
BIOMEDICAL_MODELS = ["biomistral-7b", "openbiollm-8b", "medgemma-27b"]

MODEL_DISPLAY = {
    "gpt-4o":              "GPT-4o",
    "claude-sonnet-4.6":   "Claude S4.6",
    "gemini-2.5-flash":    "Gemini 2.5F",
    "deepseek-v3":         "DeepSeek-V3",
    "qwen3-235b":          "Qwen3-235B",
    "qwen3-32b":           "Qwen3-32B",
    "llama-3.3-70b":       "Llama3.3-70B",
    "mistral-7b-instruct": "Mistral-7B",
    "gemma-3-27b":         "Gemma-27B",
    "llama-3-8b-instruct": "Llama3-8B",
    "openbiollm-8b":       "OpenBioLLM-8B",
    "biomistral-7b":       "BioMistral-7B",
    "medgemma-27b":        "MedGemma-27B",
}

MODEL_CATEGORY = {}
for m in FRONTIER_MODELS:   MODEL_CATEGORY[m] = "frontier"
for m in GENERAL_MODELS:    MODEL_CATEGORY[m] = "general"
for m in BIOMEDICAL_MODELS: MODEL_CATEGORY[m] = "biomedical"

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    "#555555",
    "axes.linewidth":    0.9,
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.7,
    "grid.linestyle":    "-",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})


def _save(fig, stem, figures_folder):
    for ext in ("png", "pdf"):
        p = f"{figures_folder}/{stem}.{ext}"
        fig.savefig(p, facecolor="white")
        print(f"Saved {p}")
    plt.close(fig)


def _fmt(ax):
    ax.tick_params(axis="both", length=3, width=0.8, colors="#444444")
    ax.set_axisbelow(True)


def load_data(results_folder, data_file, tier_file):
    with open(data_file) as f:
        dataset = json.load(f)
    for item in dataset:
        if "id" not in item and "original_index" in item:
            item["id"] = item["original_index"]
    dataset_df = pd.DataFrame(dataset)

    tier_df  = pd.read_csv(tier_file)
    tier_df["item_id"] = tier_df["item_id"].astype(int)
    tier_map = dict(zip(tier_df["item_id"], tier_df["tier_aggregate"]))
    dataset_df["tier"] = dataset_df["id"].map(tier_map)

    exclude      = {"deepseek-reasoner", "o1", "qwq-32b", "r1-distill-qwen-32b"}
    result_files = [
        f for f in glob.glob(f"{results_folder}/results_*.jsonl")
        if not any(ex in f for ex in exclude)
        and os.path.getsize(f) > 100000
    ]
    rows = []
    for rf in result_files:
        with open(rf) as f:
            for line in f:
                try:
                    rows.append(json.loads(line.strip()))
                except Exception:
                    pass

    results_df = pd.DataFrame(rows)
    results_df["is_correct"] = results_df["is_correct"].astype(bool)
    results_df["item_id"]    = results_df["item_id"].astype(int)
    results_df = results_df.drop(columns=["tier"], errors="ignore")
    results_df["tier"]     = results_df["item_id"].map(tier_map)
    results_df["category"] = results_df["model"].map(MODEL_CATEGORY)

    return dataset_df, results_df, tier_map


def figure1_finetuning_paradox(figures_folder):
    pairs = {
        "(a) 7B Scale": {
            "general":    {"name": "Mistral-7B",    "accs": [86.43, 48.99, 28.44,  3.98], "color": "#1565C0"},
            "biomedical": {"name": "BioMistral-7B", "accs": [77.25, 43.01, 33.53, 11.23], "color": "#0D47A1"},
        },
        "(b) 8B Scale": {
            "general":    {"name": "Llama-3-8B",    "accs": [85.20, 45.09, 19.96,  1.77], "color": "#2E7D32"},
            "biomedical": {"name": "OpenBioLLM-8B", "accs": [88.88, 52.69, 27.65,  4.09], "color": "#1B5E20"},
        },
        "(c) 27B Scale": {
            "general":    {"name": "Gemma-27B",    "accs": [91.27, 51.14, 20.47, 2.05], "color": "#6A1B9A"},
            "biomedical": {"name": "MedGemma-27B", "accs": [90.24, 51.01, 24.36, 2.82], "color": "#4A148C"},
        },
    }

    tiers       = ["T1", "T2", "T3", "T4"]
    tier_labels = ["T1\n(Easy)", "T2\n(Med)", "T3\n(Hard)", "T4\n(V.Hard)"]
    x           = np.arange(len(tiers))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)

    for ax, (scale_name, data) in zip(axes, pairs.items()):
        gen = data["general"]
        bio = data["biomedical"]

        ax.plot(x, gen["accs"], color=gen["color"], linestyle="-",
                linewidth=2.2, marker="o", markersize=7, label=gen["name"])
        ax.plot(x, bio["accs"], color=bio["color"], linestyle="--",
                linewidth=2.2, marker="^", markersize=7, label=bio["name"])

        gen_arr = np.array(gen["accs"])
        bio_arr = np.array(bio["accs"])
        diff    = bio_arr - gen_arr

        for i in range(len(tiers) - 1):
            if diff[i] > 0 and diff[i + 1] > 0:
                ax.axvspan(i - 0.3, i + 0.7, alpha=0.07, color=bio["color"], zorder=0)

        ax.axhline(25, color=COLORS["random"], linestyle="-.", linewidth=1.0, alpha=0.6,
                   label="Random (25%)")

        ax.set_xticks(x)
        ax.set_xticklabels(tier_labels, fontsize=9)
        ax.set_title(scale_name, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8.5, loc="upper right")

        for tier_idx in [2, 3]:
            gap = diff[tier_idx]
            if abs(gap) > 0.5:
                y_mid = (gen_arr[tier_idx] + bio_arr[tier_idx]) / 2
                sign  = "+" if gap > 0 else ""
                ax.text(tier_idx + 0.15, y_mid, f"{sign}{gap:.1f}",
                        fontsize=8, color="#555555", va="center", fontweight="bold")

    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Difficulty Tier")

    fig.tight_layout()
    _save(fig, "figure1_finetuning_paradox", figures_folder)


def figure2_accuracy_cooccurrence(results_df, figures_folder):
    tiers       = ["T1", "T2", "T3", "T4"]
    tier_labels = ["T1\n(Easy)", "T2\n(Medium)", "T3\n(Hard)", "T4\n(Very Hard)"]

    all_models    = FRONTIER_MODELS + GENERAL_MODELS + BIOMEDICAL_MODELS
    model_tier_acc = {}
    for model in all_models:
        mdf  = results_df[results_df["model"] == model]
        accs = []
        for tier in tiers:
            tdf = mdf[mdf["tier"] == tier]
            accs.append(tdf["is_correct"].mean() * 100 if len(tdf) > 0 else np.nan)
        model_tier_acc[model] = accs

    cooc_data = {
        "GPT-4o":        {"mcq": 43.3, "oe": 13.9},
        "Claude S4.6":   {"mcq": 50.6, "oe": 15.1},
        "Llama-3.3-70B": {"mcq": 38.7, "oe": 11.2},
        "Llama-3-8B":    {"mcq": 35.0, "oe":  6.5},
    }
    cooc_models = list(cooc_data.keys())
    mcq_acc     = [cooc_data[m]["mcq"] for m in cooc_models]
    oe_acc      = [cooc_data[m]["oe"]  for m in cooc_models]
    drops       = [cooc_data[m]["mcq"] - cooc_data[m]["oe"] for m in cooc_models]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 6),
                                     gridspec_kw={"width_ratios": [1.4, 1.0]})

    x = np.arange(len(tiers))

    line_styles = {
        "frontier":   {"lw": 2.5, "ls": "-",  "marker": "o", "ms": 7},
        "general":    {"lw": 1.8, "ls": "--", "marker": "s", "ms": 5},
        "biomedical": {"lw": 1.8, "ls": ":",  "marker": "^", "ms": 6},
    }
    category_colors = {
        "frontier":   ["#0D47A1", "#1565C0", "#1976D2", "#1E88E5", "#42A5F5"],
        "general":    ["#1B5E20", "#2E7D32", "#388E3C", "#43A047", "#66BB6A"],
        "biomedical": ["#4A148C", "#6A1B9A", "#7B1FA2"],
    }

    model_handles = []
    for cat, cat_models in [("frontier",   FRONTIER_MODELS),
                             ("general",    GENERAL_MODELS),
                             ("biomedical", BIOMEDICAL_MODELS)]:
        for i, model in enumerate(cat_models):
            if model not in model_tier_acc:
                continue
            accs  = model_tier_acc[model]
            color = category_colors[cat][i]
            style = line_styles[cat]
            line, = ax_a.plot(x, accs,
                              color=color, linewidth=style["lw"],
                              linestyle=style["ls"], marker=style["marker"],
                              markersize=style["ms"], alpha=0.92,
                              label=MODEL_DISPLAY.get(model, model), zorder=4)
            model_handles.append(line)

    random_line = Line2D([0], [0], color="#9E9E9E", linestyle="-.", linewidth=1.5,
                         label="Random (25%)")
    ax_a.axhline(25, color="#9E9E9E", linestyle="-.", linewidth=1.5, alpha=0.8, zorder=2)
    model_handles.append(random_line)

    ax_a.axvspan(2.5, 3.5, alpha=0.05, color="#B71C1C", zorder=1)

    claude_t3 = model_tier_acc["claude-sonnet-4.6"][2]
    gpt_t3    = model_tier_acc["gpt-4o"][2]
    ax_a.annotate(f"Claude T3 advantage\n+{claude_t3 - gpt_t3:.1f} pp vs GPT-4o",
                  xy=(2, claude_t3), xytext=(2.3, claude_t3 + 10),
                  fontsize=8, color="#0D47A1",
                  arrowprops=dict(arrowstyle="->", color="#0D47A1", lw=1.2),
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD",
                            edgecolor="#1565C0", alpha=0.9))

    bio_t4 = model_tier_acc["biomistral-7b"][3]
    ax_a.annotate(f"BioMistral-7B\nhighest T4 ({bio_t4:.1f}%)",
                  xy=(3, bio_t4), xytext=(2.55, bio_t4 + 8),
                  fontsize=8, color="#4A148C",
                  arrowprops=dict(arrowstyle="->", color="#4A148C", lw=1.2),
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5",
                            edgecolor="#6A1B9A", alpha=0.9))

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(tier_labels, fontsize=11)
    ax_a.set_xlim(-0.4, 3.4)
    ax_a.set_ylabel("Accuracy (%)", fontsize=11)
    ax_a.set_xlabel("Difficulty Tier", fontsize=11)
    ax_a.set_ylim(0, 108)
    ax_a.yaxis.grid(True, linestyle="--", alpha=0.25, zorder=1)
    ax_a.set_axisbelow(True)
    ax_a.text(0.02, 0.98, "(a)", transform=ax_a.transAxes,
              fontsize=13, fontweight="bold", va="top")

    cat_legend = [
        Line2D([0], [0], color="#1565C0", ls="-",  lw=2.5, marker="o", ms=6,
               label="── Frontier models"),
        Line2D([0], [0], color="#2E7D32", ls="--", lw=1.8, marker="s", ms=5,
               label="- - General models"),
        Line2D([0], [0], color="#6A1B9A", ls=":",  lw=1.8, marker="^", ms=6,
               label="··· Biomedical models"),
    ]
    cat_leg = ax_a.legend(handles=cat_legend, loc="lower left",
                          bbox_to_anchor=(0.01, 0.01), fontsize=8.0, framealpha=0.9)
    ax_a.add_artist(cat_leg)
    ax_a.legend(handles=model_handles,
                loc="upper right", bbox_to_anchor=(0.99, 0.99),
                fontsize=7.5, framealpha=0.9, title="Models", title_fontsize=8.0,
                ncol=2)

    xc    = np.arange(len(cooc_models))
    width = 0.35

    bars_mcq = ax_b.bar(xc - width / 2, mcq_acc, width,
                        color=COLORS["mcq"], label="MCQ Format",
                        edgecolor="white", linewidth=0.8, zorder=3)
    bars_oe  = ax_b.bar(xc + width / 2, oe_acc, width,
                        color=COLORS["openended"], label="Open-Ended Format",
                        edgecolor="white", linewidth=0.8, zorder=3)

    ax_b.axhline(25, color=COLORS["random"], linestyle="--",
                 linewidth=1.5, alpha=0.8, label="Random Chance (25%)", zorder=2)

    for i, (m, d) in enumerate(zip(cooc_models, drops)):
        x_mid = (xc[i] - width / 2 + xc[i] + width / 2) / 2 + 0.02
        ax_b.annotate("",
                      xy=(x_mid, oe_acc[i] + 1),
                      xytext=(x_mid, mcq_acc[i] - 1),
                      arrowprops=dict(arrowstyle="->", color="#D32F2F",
                                      lw=1.8, mutation_scale=14))
        ax_b.text(x_mid + 0.13, (mcq_acc[i] + oe_acc[i]) / 2,
                  f"−{d:.1f} pp",
                  color="#D32F2F", fontsize=9, fontweight="bold", va="center")

    for bar in bars_mcq:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                  f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5,
                  color="#1565C0")

    for bar in bars_oe:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                  f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5,
                  color="#1565C0")

    avg_drop = np.mean(drops)
    ax_b.text(0.98, 0.06, f"Avg drop: −{avg_drop:.1f} pp",
              transform=ax_b.transAxes, ha="right", va="bottom",
              fontsize=10, fontweight="bold", color="#D32F2F",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE",
                        edgecolor="#D32F2F", alpha=0.9))

    ax_b.set_xticks(xc)
    ax_b.set_xticklabels(cooc_models, fontsize=10)
    ax_b.set_ylabel("Accuracy (%)")
    ax_b.set_xlabel("Model")
    ax_b.set_ylim(0, 70)
    ax_b.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=1)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="upper right", fontsize=9.5)
    ax_b.text(0.02, 0.98, "(b)", transform=ax_b.transAxes,
              fontsize=13, fontweight="bold", va="top")

    fig.tight_layout(w_pad=3.0)
    _save(fig, "figure2_accuracy_cooccurrence", figures_folder)


def figure3_crossing_count(dataset_df, figures_folder):
    domain_map = {
        "enzyme_kinetics":   "Enzyme\nKinetics",
        "disease_mechanism": "Disease\nMechanisms",
        "metabolic_pathway": "Metabolic\nPathways",
        "pathway_link":      "Metabolic\nPathways",
    }

    def parse_domain(node_string):
        m = re.search(r"\[(\w+)\]", node_string)
        if not m:
            return None
        return domain_map.get(m.group(1))

    transition_counts = {}
    for _, row in dataset_df.iterrows():
        path_text = row.get("path_text", "")
        if not path_text or not isinstance(path_text, str):
            continue
        for line in path_text.strip().split("\n"):
            parts = re.split(r"--\w+-+>", line.strip())
            if len(parts) != 2:
                continue
            src = parse_domain(parts[0].strip())
            tgt = parse_domain(parts[1].strip())
            if src and tgt and src != tgt:
                transition_counts[(src, tgt)] = transition_counts.get((src, tgt), 0) + 1

    flows = sorted(transition_counts.items(), key=lambda x: -x[1])
    flows = [(s, t, v) for (s, t), v in flows]

    node_colors = {
        "Enzyme\nKinetics":    "#1565C0",
        "Disease\nMechanisms": "#B71C1C",
        "Metabolic\nPathways": "#2E7D32",
    }
    node_pos = {
        "Enzyme\nKinetics":    np.array([0.50, 0.20]),
        "Disease\nMechanisms": np.array([0.22, 0.72]),
        "Metabolic\nPathways": np.array([0.78, 0.72]),
    }
    offsets = {
        ("Enzyme\nKinetics",    "Disease\nMechanisms"):  1,
        ("Disease\nMechanisms", "Enzyme\nKinetics"):      1,
        ("Enzyme\nKinetics",    "Metabolic\nPathways"): -1,
        ("Metabolic\nPathways", "Enzyme\nKinetics"):    -1,
    }

    fig = plt.figure(figsize=(14, 5.5))
    ax_a = fig.add_axes([0.04, 0.11, 0.55, 0.80])
    ax_b = fig.add_axes([0.63, 0.02, 0.37, 0.96])

    tiers     = ["T1", "T2", "T3", "T4"]
    crossings = sorted(dataset_df["crossing_count"].unique())
    bottom    = np.zeros(len(crossings))

    for tier in tiers:
        counts = [len(dataset_df[(dataset_df["crossing_count"] == c) &
                                  (dataset_df["tier"] == tier)]) for c in crossings]
        ax_a.bar(crossings, counts, bottom=bottom,
                 color=COLORS[tier], label=tier, edgecolor="white", linewidth=0.5)
        bottom += np.array(counts)

    for i, c in enumerate(crossings):
        total = int(bottom[i])
        if total > 0:
            ax_a.text(c, total + 15, str(total), ha="center", va="bottom",
                      fontsize=8, color="#333333")

    ax_a.set_xlabel("Number of Cross-Domain Crossings")
    ax_a.set_ylabel("Number of Items")
    ax_a.set_xticks(crossings)
    ax_a.legend(title="Difficulty Tier", loc="upper right",
                handles=[mp.Patch(color=COLORS[t], label=t) for t in tiers])
    ax_a.set_ylim(0, bottom.max() * 1.12)
    ax_a.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_a.set_axisbelow(True)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.text(0.02, 0.98, "(a)", transform=ax_a.transAxes,
              fontsize=13, fontweight="bold", va="top")

    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis("off")

    node_r  = 0.145
    max_val = max(v for _, _, v in flows)
    min_val = min(v for _, _, v in flows)

    def draw_edge(ax, p1, p2, val, color, lw, offset_sign):
        mid  = (p1 + p2) / 2
        perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
        perp = perp / np.linalg.norm(perp)
        ctrl = mid + perp * 0.15 * offset_sign

        def shrink(center, toward, r):
            d = toward - center
            return center + d / np.linalg.norm(d) * r

        p1s = shrink(p1, ctrl, node_r)
        p2e = shrink(p2, ctrl, node_r)

        path  = MplPath([p1s, ctrl, p2e],
                        [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        arrow = mp.FancyArrowPatch(
            path=path,
            arrowstyle=mp.ArrowStyle.Simple(
                head_width=9 + lw * 1.0,
                head_length=11,
                tail_width=lw * 2.5,
            ),
            color=color, alpha=0.80, zorder=2, linewidth=0,
        )
        ax.add_patch(arrow)

        t  = 0.5
        lx = (1 - t) ** 2 * p1s[0] + 2 * (1 - t) * t * ctrl[0] + t ** 2 * p2e[0]
        ly = (1 - t) ** 2 * p1s[1] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * p2e[1]
        ax.text(lx, ly, f"{val:,}",
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="#111111", zorder=5,
                bbox=dict(facecolor="white", edgecolor="#cccccc",
                          linewidth=0.6, alpha=0.92, pad=2.5,
                          boxstyle="round,pad=0.3"))

    for src, tgt, val in flows:
        if src not in node_pos or tgt not in node_pos:
            continue
        lw          = 2.0 + 6.5 * (val - min_val) / (max_val - min_val)
        offset_sign = offsets.get((src, tgt), 1)
        draw_edge(ax_b, node_pos[src].copy(), node_pos[tgt].copy(),
                  val, node_colors.get(src, "#555555"), lw, offset_sign)

    for name, pos in node_pos.items():
        ax_b.add_patch(plt.Circle(pos, node_r + 0.008,
                                   color="#bbbbbb", zorder=3, linewidth=0))
        ax_b.add_patch(plt.Circle(pos, node_r,
                                   color=node_colors.get(name, "#555555"),
                                   zorder=4, linewidth=2.5, ec="white"))
        ax_b.text(pos[0], pos[1], name,
                  ha="center", va="center",
                  fontsize=10.5, fontweight="bold", color="white",
                  zorder=5, multialignment="center", linespacing=1.4)

    ax_b.text(0.01, 0.99, "(b)", transform=ax_b.transAxes,
              fontsize=13, fontweight="bold", va="top")

    _save(fig, "figure3_crossing_count", figures_folder)


if __name__ == "__main__":
    results_folder = sys.argv[1]
    data_file      = sys.argv[2]
    tier_file      = sys.argv[3]
    figures_folder = sys.argv[4] if len(sys.argv) > 4 else "figures_output"

    Path(figures_folder).mkdir(parents=True, exist_ok=True)

    dataset_df, results_df, tier_map = load_data(results_folder, data_file, tier_file)

    figure1_finetuning_paradox(figures_folder)
    figure2_accuracy_cooccurrence(results_df, figures_folder)
    figure3_crossing_count(dataset_df, figures_folder)

    print("\nAll figures saved.")