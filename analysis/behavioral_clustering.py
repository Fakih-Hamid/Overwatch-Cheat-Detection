from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BEHAVIORAL_FEATURES: Tuple[str, ...] = (
    "headshot_rate",
    "accuracy",
    "kd_ratio",
    "actions_per_minute",
    "survival_rate",
    "objective_time",
    "win_rate",
    "report_count",
    "suspicious_flags",
)


@dataclass
class ClusterResult:
    assignments: pd.DataFrame
    model: KMeans
    scaler: StandardScaler


def prepare_behavioral_matrix(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    grouped = dataset.groupby("player_id")[list(BEHAVIORAL_FEATURES)].mean()
    scaler = StandardScaler()
    matrix = scaler.fit_transform(grouped)
    return grouped, matrix, scaler


def cluster_play_styles(dataset: pd.DataFrame, n_clusters: int = 6, random_state: int = 42) -> ClusterResult:
    features, matrix, scaler = prepare_behavioral_matrix(dataset)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(matrix)
    assignments = (
        features.reset_index()
        .assign(cluster=labels)
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    return ClusterResult(assignments=assignments, model=model, scaler=scaler)


def describe_clusters(cluster_result: ClusterResult) -> pd.DataFrame:
    summary = (
        cluster_result.assignments.groupby("cluster")
        .agg(
            headshot_rate_mean=("headshot_rate", "mean"),
            accuracy_mean=("accuracy", "mean"),
            kd_ratio_mean=("kd_ratio", "mean"),
            suspicious_flags_mean=("suspicious_flags", "mean"),
            report_count_mean=("report_count", "mean"),
            win_rate_mean=("win_rate", "mean"),
        )
        .round(3)
    )
    return summary


def detect_consistency_anomalies(dataset: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    grouped = dataset.sort_values("match_id").groupby("player_id")
    metrics = []
    for player_id, matches in grouped:
        deltas = matches[["headshot_rate", "accuracy", "kd_ratio", "actions_per_minute"]].diff().dropna()
        if deltas.empty:
            continue
        stability = deltas.abs().mean()
        metrics.append(
            {
                "player_id": player_id,
                "headshot_shift": stability["headshot_rate"],
                "accuracy_shift": stability["accuracy"],
                "kd_shift": stability["kd_ratio"],
                "apm_shift": stability["actions_per_minute"],
            }
        )
    stability_df = pd.DataFrame(metrics)
    if stability_df.empty:
        return stability_df
    numeric_cols = ["headshot_shift", "accuracy_shift", "kd_shift", "apm_shift"]
    std = stability_df[numeric_cols].std(ddof=0).replace(0, np.nan)
    mean = stability_df[numeric_cols].mean()
    z_scores = (stability_df[numeric_cols] - mean) / std
    low_variance_mask = z_scores <= (-z_threshold)
    anomaly_mask = low_variance_mask.any(axis=1)
    return stability_df.loc[anomaly_mask].fillna(0.0)


def detect_rapid_improvement(dataset: pd.DataFrame, min_delta: float = 0.25) -> pd.DataFrame:
    grouped = dataset.sort_values("match_id").groupby("player_id")
    findings: List[Dict[str, float]] = []
    for player_id, matches in grouped:
        rolling_win = matches["win_rate"].rolling(window=5, min_periods=1).mean()
        rolling_kd = matches["kd_ratio"].rolling(window=5, min_periods=1).mean()
        win_jump = rolling_win.iloc[-1] - rolling_win.iloc[0]
        kd_jump = rolling_kd.iloc[-1] - rolling_kd.iloc[0]
        if win_jump >= min_delta * 100 or kd_jump >= min_delta:
            findings.append(
                {
                    "player_id": player_id,
                    "win_rate_delta": win_jump,
                    "kd_delta": kd_jump,
                    "matches_tracked": len(matches),
                }
            )
    return pd.DataFrame(findings)

