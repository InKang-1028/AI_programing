# Make the paper figures and table from the CSV files written by 04_evaluate_all_models.py.
#   results/fig_rq1_imbalance.png      RQ1: per-class accuracy vs. number of roses training images
#   results/fig_rq2_label_noise.png    RQ2: accuracy and macro-F1 vs. label-noise rate
#   results/confusion/<model>.png      confusion matrix of each model
#   results/table_results.tex          LaTeX results table

import csv  # CSV input.
import os  # File and folder operations.

import matplotlib
matplotlib.use("Agg")  # Save figures without a display.
import matplotlib.pyplot as plt
import numpy as np  # Array operations.

classes = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
minority_class = "roses"

# (model folder, value on the x-axis) for each research question.
rq1_runs = [("base", 200), ("imb_roses50", 50), ("imb_roses20", 20), ("imb_roses5", 5)]  # roses images
rq2_runs = [("base", 0), ("noise10", 10), ("noise30", 30), ("noise50", 50)]               # flipped labels (%)

result_dir = "./results"

# Read the per-model accuracy table.
with open(os.path.join(result_dir, "accuracy.csv")) as f:
    acc = {row["model"]: row for row in csv.DictReader(f)}

# Read confusion matrices: cms[model][true index, predicted index] = count.
cms = {}
with open(os.path.join(result_dir, "confusion_matrix.csv")) as f:
    for row in csv.DictReader(f):
        cm = cms.setdefault(row["model"], np.zeros((len(classes), len(classes)), dtype=int))
        cm[classes.index(row["true"]), classes.index(row["pred"])] = int(row["count"])

# ---- RQ1: class imbalance
runs = [(m, n) for m, n in rq1_runs if m in acc]
if len(runs) >= 2:
    x = [n for _, n in runs]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for cls in classes:
        y = [float(acc[m][f"{cls}_acc(%)"]) for m, _ in runs]
        style = {"linewidth": 2.5, "marker": "o", "zorder": 3} if cls == minority_class else \
                {"linewidth": 1, "marker": ".", "alpha": 0.6}
        ax.plot(x, y, label=cls, **style)
    ax.plot(x, [float(acc[m]["overall_acc(%)"]) for m, _ in runs], "k--", label="overall")
    ax.set_xscale("log")
    ax.set_xticks(x, [str(v) for v in x])
    ax.invert_xaxis()  # Fewer roses images to the right.
    ax.set_xlabel(f"{minority_class} training images (other classes: 200)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "fig_rq1_imbalance.png"), dpi=200)
    plt.close(fig)
    print("Saved: fig_rq1_imbalance.png")

# ---- RQ2: label noise
runs = [(m, r) for m, r in rq2_runs if m in acc]
if len(runs) >= 2:
    x = [r for _, r in runs]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(x, [float(acc[m]["overall_acc(%)"]) for m, _ in runs], "o-", label="accuracy")
    ax.plot(x, [float(acc[m]["macro_f1(%)"]) for m, _ in runs], "s--", label="macro-F1")
    ax.set_xticks(x)
    ax.set_xlabel("Training labels flipped (%)")
    ax.set_ylabel("Test score (%)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "fig_rq2_label_noise.png"), dpi=200)
    plt.close(fig)
    print("Saved: fig_rq2_label_noise.png")

# ---- Confusion matrices
os.makedirs(os.path.join(result_dir, "confusion"), exist_ok=True)
for model, cm in cms.items():
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(axis=1).max())
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(model)
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "confusion", f"{model}.png"), dpi=200)
    plt.close(fig)
print(f"Saved: confusion/ ({len(cms)} models)")

# ---- Paper figure: three panels sized for the full width of a two-column page (7 in).
#   (a) RQ1 per-class accuracy, (b) RQ2 accuracy with per-class range, (c) where roses test images go.
colors = dict(zip(classes, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
markers = dict(zip(classes, ["o", "s", "D", "^", "v"]))
rq1 = [(m, n) for m, n in rq1_runs if m in acc]
rq2 = [(m, r) for m, r in rq2_runs if m in acc]
if len(rq1) == len(rq1_runs) and len(rq2) == len(rq2_runs):
    plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.95), gridspec_kw={"width_ratios": [1, 1, 1.15]})

    # (a) RQ1: per-class accuracy as roses shrinks (evenly spaced x positions).
    ax = axes[0]
    xs = range(len(rq1))
    for cls in classes:
        y = [float(acc[m][f"{cls}_acc(%)"]) for m, _ in rq1]
        bold = cls == minority_class
        ax.plot(xs, y, color=colors[cls], marker=markers[cls], markersize=4 if bold else 3,
                linewidth=2 if bold else 1, alpha=1 if bold else 0.8, label=cls, zorder=3 if bold else 2)
    ax.plot(xs, [float(acc[m]["overall_acc(%)"]) for m, _ in rq1], "k--", linewidth=1, label="overall")
    ax.set_xticks(list(xs), [str(n) for _, n in rq1])
    ax.set_xlabel("roses training images (others: 200)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("(a) RQ1: class imbalance", fontsize=8)

    # (b) RQ2: overall accuracy, with the range across the five classes as a band.
    ax = axes[1]
    x = [r for _, r in rq2]
    per_class = np.array([[float(acc[m][f"{c}_acc(%)"]) for c in classes] for m, _ in rq2])
    ax.fill_between(x, per_class.min(axis=1), per_class.max(axis=1), color="#2a78d6", alpha=0.15,
                    linewidth=0, label="per-class range")
    ax.plot(x, [float(acc[m]["overall_acc(%)"]) for m, _ in rq2], "k-o", markersize=3.5, linewidth=1.5,
            label="overall")
    ax.plot(x, [float(acc[m]["macro_f1(%)"]) for m, _ in rq2], color="#eb6834", marker="s", markersize=3,
            linewidth=1, linestyle=":", label="macro-F1")
    ax.axhline(100 / len(classes), color="gray", linewidth=0.8, linestyle="--")
    ax.text(x[-1], 100 / len(classes) + 2, "chance", ha="right", color="gray")
    ax.set_xticks(x)
    ax.set_xlabel("Training labels flipped (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.2), frameon=False)
    ax.set_title("(b) RQ2: label noise", fontsize=8)

    # (c) Predictions for the 100 roses test images, per RQ1 model (stacked bars).
    ax = axes[2]
    for i, (m, n) in enumerate(rq1):
        left = 0
        for cls in classes:
            count = cms[m][classes.index(minority_class), classes.index(cls)]
            if count:
                ax.barh(i, count, left=left, color=colors[cls], edgecolor="white", linewidth=1, height=0.7)
                if count >= 8:
                    ax.text(left + count / 2, i, str(count), ha="center", va="center", fontsize=6.5)
            left += count
    ax.set_yticks(range(len(rq1)), [f"{n} roses" for _, n in rq1])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("roses test images predicted as ...")
    ax.set_title("(c) RQ1: where roses go", fontsize=8)

    # One class legend for panels (a) and (c), below the figure (same colors in both).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handlelength=2)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(result_dir, "fig_paper.pdf"))
    fig.savefig(os.path.join(result_dir, "fig_paper.png"), dpi=200)
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    print("Saved: fig_paper.pdf")

# ---- LaTeX table: training-set composition, per-class accuracy, overall accuracy, macro-F1
# Composition of each training set (roses images, % flipped labels); see data/make_train_sets.py.
setup = {m: (n, 0) for m, n in rq1_runs}
setup.update({m: (200, r) for m, r in rq2_runs})
short = {"daisy": "Dai.", "dandelion": "Dan.", "roses": "Ros.", "sunflowers": "Sun.", "tulips": "Tul."}
lines = [
    r"\begin{table}[t]",
    r"\centering\footnotesize",
    r"\setlength{\tabcolsep}{1.7pt}",
    r"\begin{tabular}{lcc|" + "c" * len(classes) + "|cc}",
    r"\toprule",
    r"& \multicolumn{2}{c|}{Training set} & \multicolumn{" + str(len(classes))
    + r"}{c|}{Per-class accuracy} & & \\",
    r"Model & \#Ros. & Noise & " + " & ".join(short[c] for c in classes) + r" & Acc. & F1 \\",
    r"\midrule",
]
for model, row in acc.items():
    n_roses, noise = setup[model]
    cells = [str(n_roses), str(noise)]
    cells += [f"{float(row[f'{c}_acc(%)']):.0f}" for c in classes]
    cells += [f"{float(row['overall_acc(%)']):.1f}", f"{float(row['macro_f1(%)']):.1f}"]
    if model == "imb_roses5":
        cells[2 + classes.index(minority_class)] = r"\textbf{" + cells[2 + classes.index(minority_class)] + "}"
    lines.append(model.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    if model in ("base", "imb_roses5"):
        lines.append(r"\midrule")
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Training sets and test results (\%). \#Ros.: roses training images (others: 200); "
    r"Noise: \% of training labels flipped. Per-class accuracy is recall; F1 is macro-F1.}",
    r"\label{tab:results}",
    r"\end{table}",
]
with open(os.path.join(result_dir, "table_results.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("Saved: table_results.tex")
