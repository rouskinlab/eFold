import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _read_first_existing_csv(path_candidates):
    for path in path_candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"None of the CSV paths exist: {path_candidates}")


def load_all_results(base_dir):
    configs = [
        ("PDB", {
            "baseline_og": os.path.join(base_dir, "PDB_f1_baseline_og.csv"),
            "baseline_avg": os.path.join(base_dir, "PDB_f1_baseline_avg.csv"),
            "baseline_2": os.path.join(base_dir, "PDB_f1_baseline_avg_2.csv"),
            "abl1": os.path.join(base_dir, "PDB_f1_ablation1_simple.csv"),
            "abl1_less": os.path.join(base_dir, "PDB_f1_ablation1_simple_lessparams.csv"),
            "abl2": os.path.join(base_dir, "PDB_f1_ablation2_2.csv"),
        }),
        ("ArchiveII", {
            "baseline_og": os.path.join(base_dir, "archiveII_f1_baseline_og.csv"),
            "baseline_avg": os.path.join(base_dir, "archiveII_f1_baseline_avg.csv"),
            "baseline_2": os.path.join(base_dir, "archiveII_f1_baseline_avg_2.csv"),
            "abl1": os.path.join(base_dir, "archiveII_f1_ablation1_simple.csv"),
            "abl1_less": os.path.join(base_dir, "archiveII_f1_ablation1_simple_lessparams.csv"),
            "abl2": os.path.join(base_dir, "archiveII_f1_ablation2_2.csv"),
        }),
        ("IncRNA", {
            "baseline_og": os.path.join(base_dir, "lncRNA_nonFiltered_f1_baseline_og.csv"),
            "baseline_avg": os.path.join(base_dir, "lncRNA_nonFiltered_f1_baseline_avg.csv"),
            "baseline_2": os.path.join(base_dir, "lncRNA_nonFiltered_f1_baseline_avg_2.csv"),
            "abl1": os.path.join(base_dir, "lncRNA_nonFiltered_f1_ablation1_simple.csv"),
            "abl1_less": os.path.join(base_dir, "lncRNA_nonFiltered_f1_ablation1_simple_lessparams.csv"),
            "abl2": os.path.join(base_dir, "lncRNA_nonFiltered_f1_ablation2_2.csv"),
        }),
        ("viral_fragments", {
            "baseline_og": os.path.join(base_dir, "viral_fragments_f1_baseline_og.csv"),
            "baseline_avg": os.path.join(base_dir, "viral_fragments_f1_baseline_avg.csv"),
            "baseline_2": os.path.join(base_dir, "viral_fragments_f1_baseline_avg_2.csv"),
            "abl1": os.path.join(base_dir, "viral_fragments_f1_ablation1_simple.csv"),
            "abl1_less": os.path.join(base_dir, "viral_fragments_f1_ablation1_simple_lessparams.csv"),
            "abl2": os.path.join(base_dir, "viral_fragments_f1_ablation2_2.csv"),
        }),
    ]

    records = []
    for dataset, cfg in configs:
        baseline_og_df = pd.read_csv(cfg["baseline_og"])\
            .assign(dataset=dataset, model="Baseline downloaded")
        baseline_avg_df = pd.read_csv(cfg["baseline_avg"])\
            .assign(dataset=dataset, model="Baseline reproduced")
        baseline_2_df = pd.read_csv(cfg["baseline_2"])\
            .assign(dataset=dataset, model="Baseline 2")
        abl1_df = pd.read_csv(cfg["abl1"])\
            .assign(dataset=dataset, model="Ablation 1")
        abl1_less_df = pd.read_csv(cfg["abl1_less"])\
            .assign(dataset=dataset, model="Ablation 1 (less params)")
        abl2_df = pd.read_csv(cfg["abl2"])\
            .assign(dataset=dataset, model="Ablation 2")

        records.extend([baseline_og_df, baseline_avg_df, baseline_2_df, abl1_df, abl1_less_df, abl2_df])

    df = pd.concat(records, ignore_index=True)
    if "f1" not in df.columns:
        raise ValueError("Input CSVs must contain an 'f1' column")
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    df = df.dropna(subset=["f1"])  # ensure numeric
    return df


def plot_violin(df, out_dir):
    sns.set(style="whitegrid", context="talk")
    palette = {
        "Baseline downloaded": "#1f77b4",
        "Baseline reproduced": "#9467bd",
        "Baseline 2": "#17becf",
        "Ablation 1": "#ff7f0e",
        "Ablation 1 (less params)": "#d62728",
        "Ablation 2": "#2ca02c",
    }
    dataset_order = ["PDB", "ArchiveII", "IncRNA", "viral_fragments"]
    model_order = ["Baseline downloaded", "Baseline reproduced", "Baseline 2", "Ablation 1", "Ablation 1 (less params)", "Ablation 2"]

    plt.figure(figsize=(14, 5))
    ax = sns.violinplot(
        data=df,
        x="dataset",
        y="f1",
        hue="model",
        order=dataset_order,
        hue_order=model_order,
        palette=palette,
        cut=0,
        inner="box",
        density_norm="width",
        linewidth=0.8,
    )
    ax.set_xlabel("Dataset", fontsize=14)
    ax.set_ylabel("F1 score", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)
        lbl.set_horizontalalignment("center")
    ax.legend(
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=True,
        fontsize=11,
        title_fontsize=12,
    )
    plt.tight_layout(rect=(0, 0, 0.8, 1))

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "model_comparison_violin.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return out_png


if __name__ == "__main__":
    base = "/n/data1/hms/microbiology/rouskin/lab/projects/deep_learning/eFold/data/test_sets"
    df_all = load_all_results(base)
    output_path = plot_violin(df_all, base)
    print(f"Saved violin plot to: {output_path}")


