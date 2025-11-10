from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import seaborn as sns


def scatter_headshot_vs_reaction(dataset, output_path: Optional[Path] = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=dataset,
        x="reaction_time_ms",
        y="headshot_rate",
        hue="cheat_label",
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("Headshot Rate vs. Reaction Time")
    ax.set_xlabel("Reaction Time (ms)")
    ax.set_ylabel("Headshot Rate")
    fig.tight_layout()
    return _save_or_render(fig, output_path)


def distribution_plot(dataset, metric: str, output_path: Optional[Path] = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=dataset,
        x=metric,
        hue="cheat_label",
        element="step",
        stat="density",
        common_norm=False,
        bins=40,
        ax=ax,
    )
    ax.set_title(f"{metric.replace('_', ' ').title()} Distribution")
    fig.tight_layout()
    return _save_or_render(fig, output_path)


def cluster_visualization(assignments, output_path: Optional[Path] = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=assignments,
        x="kd_ratio",
        y="accuracy",
        hue="cluster",
        palette="viridis",
        s=80,
        ax=ax,
    )
    ax.set_title("Behavioral Cluster Map")
    fig.tight_layout()
    return _save_or_render(fig, output_path)


def roc_curve_plot(fprs, tprs, label, output_path: Optional[Path] = None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fprs, tprs, label=label)
    ax.plot([0, 1], [0, 1], "k--", label="Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Model ROC Curve")
    ax.legend()
    fig.tight_layout()
    return _save_or_render(fig, output_path)


def _save_or_render(fig, output_path: Optional[Path]):
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        return None
    return fig

