from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import zscore


@dataclass
class OutlierSummary:
    metric: str
    z_threshold: float
    outlier_count: int
    player_ids: List[str]


def aggregate_player_matches(dataset: pd.DataFrame) -> pd.DataFrame:
    grouped = dataset.groupby(["player_id", "cheat_label"])
    features = grouped.agg(
        headshot_rate_mean=("headshot_rate", "mean"),
        accuracy_mean=("accuracy", "mean"),
        kd_ratio_mean=("kd_ratio", "mean"),
        win_rate_mean=("win_rate", "mean"),
        reaction_time_ms_mean=("reaction_time_ms", "mean"),
        actions_per_minute_mean=("actions_per_minute", "mean"),
        survival_rate_mean=("survival_rate", "mean"),
        report_count_sum=("report_count", "sum"),
        suspicious_flags_sum=("suspicious_flags", "sum"),
        matches_played=("match_id", "count"),
        objective_time_mean=("objective_time", "mean"),
        damage_done_mean=("damage_done", "mean"),
        healing_done_mean=("healing_done", "mean"),
    ).reset_index()
    return features


def compute_z_score_outliers(
    player_summary: pd.DataFrame, metrics: Iterable[str], threshold: float = 3.5
) -> List[OutlierSummary]:
    summaries: List[OutlierSummary] = []
    for metric in metrics:
        values = player_summary[metric].to_numpy()
        standardized = zscore(values, nan_policy="omit")
        mask = np.abs(standardized) >= threshold
        summaries.append(
            OutlierSummary(
                metric=metric,
                z_threshold=threshold,
                outlier_count=int(mask.sum()),
                player_ids=player_summary.loc[mask, "player_id"].tolist(),
            )
        )
    return summaries


def flag_impossible_performance(player_summary: pd.DataFrame) -> pd.DataFrame:
    conditions = (
        (player_summary["headshot_rate_mean"] > 0.85)
        | (player_summary["accuracy_mean"] > 0.85)
        | (player_summary["reaction_time_ms_mean"] < 110)
        | (player_summary["kd_ratio_mean"] > 4.5)
        | (player_summary["win_rate_mean"] > 90)
    )
    return player_summary.loc[conditions].copy()


def correlation_with_reports(player_summary: pd.DataFrame) -> pd.DataFrame:
    numeric_fields = [
        "headshot_rate_mean",
        "accuracy_mean",
        "kd_ratio_mean",
        "win_rate_mean",
        "reaction_time_ms_mean",
        "actions_per_minute_mean",
        "survival_rate_mean",
        "objective_time_mean",
        "damage_done_mean",
        "healing_done_mean",
    ]
    corr = player_summary[numeric_fields + ["report_count_sum"]].corr()
    return corr["report_count_sum"].sort_values(ascending=False).to_frame(name="correlation")


def risk_band_from_zscore(value: float, threshold: float) -> str:
    if value >= threshold + 1:
        return "critical"
    if value >= threshold:
        return "high"
    if value >= threshold - 1:
        return "moderate"
    return "baseline"

