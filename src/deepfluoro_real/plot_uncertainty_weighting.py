from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METHOD_LABELS = {
    "no_weights": "No Weights",
    "discrete_selection": "Discrete Selection",
    "continuous_weighting": "Continuous Weighting",
}

METHOD_ORDER = ["no_weights", "discrete_selection", "continuous_weighting"]
PALETTE = {
    "No Weights": "#4c566a",
    "Discrete Selection": "#dd8452",
    "Continuous Weighting": "#4c9f70",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot boxplots for real-image uncertainty weighting experiments.")
    parser.add_argument(
        "--input_csv",
        default="visualizations/deepfluoro_real_uncertainty_weighting_hard_dist_dropout_01/per_case_results.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="visualizations/deepfluoro_real_uncertainty_weighting_hard_dist_dropout_01/plots",
    )
    return parser.parse_args()


def _prepare_df(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df = df[df["mtre_mm"].notna()].copy()
    df["method_label"] = df["method"].map(METHOD_LABELS)
    df["k"] = df["k"].astype(int)
    df["k_label"] = df["k"].map(lambda k: f"K={k}")
    return df


def _add_summary_box(ax: plt.Axes, values: pd.Series, unit: str, title_prefix: str) -> None:
    if values.empty:
        return
    text = (
        f"Median: {values.median():.1f} {unit}\n"
        f"IQR: {values.quantile(0.25):.1f} to {values.quantile(0.75):.1f}\n"
        f"Mean: {values.mean():.1f} {unit}"
    )
    ax.text(
        0.98,
        0.98,
        text,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.92),
    )
    ax.set_title(title_prefix, fontsize=15)


def make_boxplot_figure(df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.15)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    metric_specs = [
        ("rotation_diff_deg", "Rotation Error", "deg"),
        ("translation_diff_mm", "Translation Error", "mm"),
        ("mtre_mm", "mTRE", "mm"),
    ]

    for ax, (metric, title, unit) in zip(axes, metric_specs):
        sns.boxplot(
            data=df,
            x="k_label",
            y=metric,
            hue="method_label",
            order=[f"K={k}" for k in sorted(df["k"].unique())],
            hue_order=[METHOD_LABELS[m] for m in METHOD_ORDER if METHOD_LABELS[m] in df["method_label"].unique()],
            palette=PALETTE,
            showfliers=False,
            width=0.75,
            ax=ax,
        )
        ax.set_xlabel("Top-K Filtered Landmarks", fontsize=12)
        ax.set_ylabel(unit, fontsize=12)
        ax.tick_params(axis="x", rotation=0)
        _add_summary_box(ax, df[metric], unit, title)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, title=None, loc="upper left", fontsize=10, frameon=True)
    for ax in axes[1:]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_success_plot(df: pd.DataFrame, output_path: Path) -> None:
    success_df = (
        df.groupby(["method_label", "k"], as_index=False)
        .size()
        .rename(columns={"size": "success_cases"})
    )

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=success_df,
        x="k",
        y="success_cases",
        hue="method_label",
        style="method_label",
        markers=True,
        dashes=False,
        palette=PALETTE,
        ax=ax,
    )
    ax.set_title("Successful Pose Cases vs K", fontsize=15)
    ax.set_xlabel("Top-K Filtered Landmarks", fontsize=12)
    ax.set_ylabel("Successful Cases", fontsize=12)
    ax.set_xticks(sorted(success_df["k"].unique()))
    ax.legend(title=None, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare_df(input_csv)
    make_boxplot_figure(df, output_dir / "weighting_boxplots.png")
    make_success_plot(df, output_dir / "weighting_success_vs_k.png")


if __name__ == "__main__":
    main()
