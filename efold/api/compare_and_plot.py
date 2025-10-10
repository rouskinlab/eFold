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
            "baseline_reboot": os.path.join(base_dir, "PDB_f1_baseline_reboot.csv"),
            "abl1": os.path.join(base_dir, "PDB_f1_ablation1_reboot.csv"),
            "abl1_less": os.path.join(base_dir, "PDB_f1_ablation1_simple_lessparams.csv"),
            "abl2": os.path.join(base_dir, "PDB_f1_ablation2_2.csv"),
        }),
        ("ArchiveII", {
            "baseline_og": os.path.join(base_dir, "archiveII_f1_baseline_og.csv"),
            "baseline_avg": os.path.join(base_dir, "archiveII_f1_baseline_avg.csv"),
            "baseline_reboot": os.path.join(base_dir, "archiveII_f1_baseline_reboot.csv"),
            "abl1": os.path.join(base_dir, "archiveII_f1_ablation1_reboot.csv"),
            "abl1_less": os.path.join(base_dir, "archiveII_f1_ablation1_simple_lessparams.csv"),
            "abl2": os.path.join(base_dir, "archiveII_f1_ablation2_2.csv"),
        }),
        ("IncRNA", {
            "baseline_og": os.path.join(base_dir, "lncRNA_nonFiltered_f1_baseline_og.csv"),
            "baseline_avg": os.path.join(base_dir, "lncRNA_nonFiltered_f1_baseline_avg.csv"),
            "baseline_reboot": os.path.join(base_dir, "lncRNA_nonFiltered_f1_baseline_reboot.csv"),
            "abl1": os.path.join(base_dir, "lncRNA_nonFiltered_f1_ablation1_reboot.csv"),
            "abl1_less": os.path.join(base_dir, "lncRNA_nonFiltered_f1_ablation1_simple_lessparams.csv"),
            "abl2": os.path.join(base_dir, "lncRNA_nonFiltered_f1_ablation2_2.csv"),
        }),
        ("viral_fragments", {
            "baseline_og": os.path.join(base_dir, "viral_fragments_f1_baseline_og.csv"),
            "baseline_avg": os.path.join(base_dir, "viral_fragments_f1_baseline_avg.csv"),
            "baseline_reboot": os.path.join(base_dir, "viral_fragments_f1_baseline_reboot.csv"),
            "abl1": os.path.join(base_dir, "viral_fragments_f1_ablation1_reboot.csv"),
            "abl1_less": os.path.join(base_dir, "viral_fragments_f1_ablation1_simple_lessparams.csv"),
            "abl2": os.path.join(base_dir, "viral_fragments_f1_ablation2_2.csv"),
        }),
    ]

    records = []
    for dataset, cfg in configs:
        baseline_og_df = pd.read_csv(cfg["baseline_og"])\
            .assign(dataset=dataset, model="Baseline downloaded")
        baseline_reboot_df = pd.read_csv(cfg["baseline_reboot"])\
            .assign(dataset=dataset, model="Baseline reboot")
        abl1_df = pd.read_csv(cfg["abl1"])\
            .assign(dataset=dataset, model="Ablation 1 reboot")

        records.extend([baseline_og_df, baseline_reboot_df, abl1_df])

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
        "Baseline reboot": "#8c564b",
        "Ablation 1 reboot": "#ff7f0e",
    }
    dataset_order = ["PDB", "ArchiveII", "IncRNA", "viral_fragments"]
    model_order = [
        "Baseline downloaded",
        "Baseline reboot",
        "Ablation 1 reboot",
    ]

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
    # Compute mean F1 per dataset and model, print and save to text
    dataset_order = ["PDB", "ArchiveII", "IncRNA", "viral_fragments"]
    model_order = ["Baseline downloaded", "Baseline reboot", "Ablation 1 reboot"]
    means = (
        df_all.groupby(["dataset", "model"])['f1']
        .mean()
        .reset_index()
        .rename(columns={"f1": "mean_f1"})
    )
    means["dataset"] = pd.Categorical(means["dataset"], dataset_order, ordered=True)
    means["model"] = pd.Categorical(means["model"], model_order, ordered=True)
    means = means.sort_values(["dataset", "model"]).reset_index(drop=True)

    out_txt = os.path.join(base, "mean_f1_summary.txt")
    lines = ["dataset\tmodel\tmean_f1"] + [
        f"{row.dataset}\t{row.model}\t{row.mean_f1:.4f}" for row in means.itertuples(index=False)
    ]
    print("\n".join(lines))
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved mean F1 summary to: {out_txt}")
    output_path = plot_violin(df_all, base)
    print(f"Saved violin plot to: {output_path}")

