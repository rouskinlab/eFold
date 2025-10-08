import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_results(base_dir: str) -> pd.DataFrame:
    configs = [
        ("PDB", {
            "baseline_avg": os.path.join(base_dir, "PDB_f1_baseline_avg.csv"),
            "abl2": os.path.join(base_dir, "PDB_f1_ablation2_2.csv"),
        }),
        ("ArchiveII", {
            "baseline_avg": os.path.join(base_dir, "archiveII_f1_baseline_avg.csv"),
            "abl2": os.path.join(base_dir, "archiveII_f1_ablation2_2.csv"),
        }),
        ("IncRNA", {
            "baseline_avg": os.path.join(base_dir, "lncRNA_nonFiltered_f1_baseline_avg.csv"),
            "abl2": os.path.join(base_dir, "lncRNA_nonFiltered_f1_ablation2_2.csv"),
        }),
        ("viral_fragments", {
            "baseline_avg": os.path.join(base_dir, "viral_fragments_f1_baseline_avg.csv"),
            "abl2": os.path.join(base_dir, "viral_fragments_f1_ablation2_2.csv"),
        }),
    ]

    parts = []
    for dataset, cfg in configs:
        for model_name, key in [("Baseline Averaged", "baseline_avg"), ("Ablation 2", "abl2")]:
            df = pd.read_csv(cfg[key])
            df = df.assign(dataset=dataset, model=model_name)
            parts.append(df)

    df_all = pd.concat(parts, ignore_index=True)
    if "f1" not in df_all.columns:
        raise ValueError("Input CSVs must contain an 'f1' column")

    # Derive sequence length if not present
    if "length" not in df_all.columns:
        if "sequence" in df_all.columns:
            df_all["length"] = df_all["sequence"].astype(str).str.len()
        else:
            raise ValueError("Need either 'length' or 'sequence' column to compute sequence length")

    df_all["f1"] = pd.to_numeric(df_all["f1"], errors="coerce")
    df_all["length"] = pd.to_numeric(df_all["length"], errors="coerce")
    df_all = df_all.dropna(subset=["f1", "length"])  # ensure numeric
    return df_all


def plot_scatter_with_trend(df: pd.DataFrame, out_path: str) -> str:
    sns.set(style="whitegrid", context="talk")
    palette = {
        "Baseline Averaged": "#9467bd",
        "Ablation 2": "#2ca02c",
    }

    # Figure and axis
    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    # Shade long-sequence region and add threshold line
    long_threshold = 1200
    xmax = float(df["length"].max())
    ax.axvspan(long_threshold, xmax, color="#000000", alpha=0.06, zorder=0)
    ax.axvline(long_threshold, color="#7f7f7f", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(long_threshold + 10, 0.98, "1200+", va="top", ha="left", fontsize=11, color="#7f7f7f")

    # Plot Ablation 2 points first (under), then Baseline Averaged points on top for visibility
    for model, z in [("Ablation 2", 2), ("Baseline Averaged", 3)]:
        sub = df[df["model"] == model]
        sns.scatterplot(
            data=sub,
            x="length",
            y="f1",
            color=palette[model],
            alpha=0.35 if model == "Ablation 2" else 0.5,
            s=18 if model == "Ablation 2" else 22,
            linewidth=0.2,
            edgecolor="none",
            ax=ax,
            legend=False,
            zorder=z,
        )

    # Binned median trend lines per model
    num_bins = 30
    bins = pd.interval_range(start=df["length"].min(), end=xmax, periods=num_bins)
    df["len_bin"] = pd.cut(df["length"], bins=bins, include_lowest=True)
    grouped = df.groupby(["model", "len_bin"])['f1'].median().reset_index()
    grouped["len_mid"] = grouped["len_bin"].apply(lambda iv: (iv.left + iv.right) / 2)
    for model in ["Baseline Averaged", "Ablation 2"]:
        color = palette[model]
        sub = grouped[grouped["model"] == model].sort_values("len_mid")
        ax.plot(sub["len_mid"], sub["f1"], color=color, linewidth=2.2, label=model, zorder=4)

    # Compute and annotate collapse percentage for Ablation 2 in long region
    long = df[(df["length"] >= long_threshold) & (df["model"] == "Ablation 2")]
    if len(long) > 0:
        collapsed = (long["f1"] <= 0.05).mean() * 100.0
        ax.text(long_threshold + (xmax - long_threshold) * 0.02, 0.07,
                f"Ablation 2: {collapsed:.0f}% near-zero F1 for long sequences",
                fontsize=11, color=palette["Ablation 2"], ha="left", va="bottom")

    ax.set_xlabel("Sequence length", fontsize=14)
    ax.set_ylabel("F1 score", fontsize=14)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True, fontsize=11, title_fontsize=12)
    plt.tight_layout(rect=(0, 0, 0.8, 1))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


if __name__ == "__main__":
    base_dir = "/n/data1/hms/microbiology/rouskin/lab/projects/deep_learning/eFold/data/test_sets"
    output = os.path.join(base_dir, "length_vs_f1_baselineAvg_vs_ablation2_2.png")
    df_all = load_results(base_dir)
    out = plot_scatter_with_trend(df_all, output)
    print(f"Saved plot to: {out}")


