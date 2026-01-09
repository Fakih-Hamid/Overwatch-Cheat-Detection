from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURE_COLUMNS: Tuple[str, ...] = (
    "headshot_rate",
    "accuracy",
    "kd_ratio",
    "win_rate",
    "reaction_time_ms",
    "actions_per_minute",
    "position_changes",
    "survival_rate",
    "ultimate_efficiency",
    "objective_time",
    "report_count",
    "suspicious_flags",
    "account_age_days",
)


@dataclass
class CheatProbabilityResult:
    player_id: str
    cheat_probability: float
    cheat_type: str
    model_decision: float


def detect_aimbot(players_df: pd.DataFrame) -> pd.DataFrame:
    mask = (players_df["headshot_rate"] > 0.7) & (players_df["reaction_time_ms"] < 100)
    return players_df.loc[mask].copy()


def detect_wallhack(players_df: pd.DataFrame) -> pd.DataFrame:
    mask = (players_df["survival_rate"] > 0.9) & (players_df["position_changes"] < 18)
    return players_df.loc[mask].copy()


def detect_triggerbot(players_df: pd.DataFrame) -> pd.DataFrame:
    grouped = players_df.sort_values("match_id").groupby("player_id")
    results = []
    for player_id, matches in grouped:
        delta = matches["reaction_time_ms"].diff().dropna()
        if delta.empty:
            continue
        interval_variance = delta.abs().std()
        accuracy_std = matches["accuracy"].std()
        if interval_variance < 8 and (accuracy_std or 0) < 0.03:
            sample = matches.iloc[-1].copy()
            sample["interval_variance"] = interval_variance
            sample["accuracy_std"] = accuracy_std or 0.0
            results.append(sample)
    return pd.DataFrame(results)


def identify_smurf_accounts(players_df: pd.DataFrame) -> pd.DataFrame:
    grouped = players_df.groupby("player_id").agg(
        account_age_days=("account_age_days", "min"),
        kd_ratio_mean=("kd_ratio", "mean"),
        win_rate_mean=("win_rate", "mean"),
        matches_played=("match_id", "count"),
    )
    mask = (grouped["account_age_days"] < 30) & ((grouped["kd_ratio_mean"] > 2.8) | (grouped["win_rate_mean"] > 75))
    return grouped.loc[mask].reset_index()


def rank_suspicious_players(players_df: pd.DataFrame, weight_reports: float = 1.5) -> pd.DataFrame:
    summary = players_df.groupby("player_id").agg(
        headshot_rate_mean=("headshot_rate", "mean"),
        reaction_time_ms_min=("reaction_time_ms", "min"),
        kd_ratio_mean=("kd_ratio", "mean"),
        win_rate_mean=("win_rate", "mean"),
        accuracy_mean=("accuracy", "mean"),
        report_count_sum=("report_count", "sum"),
        suspicious_flags_sum=("suspicious_flags", "sum"),
    )
    score = (
        summary["headshot_rate_mean"] * 0.25
        + (100 - summary["reaction_time_ms_min"]) * 0.2 / 100
        + summary["kd_ratio_mean"] * 0.2
        + summary["win_rate_mean"] * 0.15 / 100
        + summary["accuracy_mean"] * 0.1
        + summary["suspicious_flags_sum"] * 0.05
        + summary["report_count_sum"] * weight_reports * 0.05
    )
    ranked = summary.assign(risk_score=score).sort_values("risk_score", ascending=False).reset_index()
    return ranked


def load_anomaly_model(model_path: Path) -> IsolationForest:
    artifact = joblib.load(model_path)
    if isinstance(artifact, IsolationForest):
        return artifact
    if isinstance(artifact, dict) and "isolation_forest" in artifact:
        model = artifact["isolation_forest"]
        if isinstance(model, IsolationForest):
            return model
    raise TypeError("An IsolationForest model was not found in the saved artifact.")


def get_feature_matrix(players_df: pd.DataFrame, feature_columns: Iterable[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    return players_df.groupby("player_id")[list(feature_columns)].mean().reset_index()


def calculate_cheat_probability(
    players_df: pd.DataFrame,
    model: Optional[IsolationForest] = None,
    model_path: Optional[Path] = None,
) -> pd.DataFrame:
    if model is None:
        if model_path is None:
            model_path = Path("models") / "anomaly_model.pkl"
        model = load_anomaly_model(model_path)

    feature_data = get_feature_matrix(players_df)
    feature_values = feature_data[list(FEATURE_COLUMNS)].to_numpy()
    scores = model.decision_function(feature_values)
    normalized = _scale_scores(scores)
    labels = model.predict(feature_values)

    results = []
    for idx, row in feature_data.iterrows():
        score = normalized[idx]
        predicted_label = labels[idx]
        cheat_type = _infer_cheat_type(row, predicted_label)
        results.append(
            CheatProbabilityResult(
                player_id=row["player_id"],
                cheat_probability=score,
                cheat_type=cheat_type,
                model_decision=scores[idx],
            )
        )
    return pd.DataFrame(results)


def _scale_scores(scores: np.ndarray) -> np.ndarray:
    min_score = np.min(scores)
    max_score = np.max(scores)
    if np.isclose(min_score, max_score):
        return np.ones_like(scores) * 0.5
    scaled = (scores - min_score) / (max_score - min_score)
    return scaled


def _infer_cheat_type(row: pd.Series, label: int) -> str:
    if label == -1:
        if row["headshot_rate"] > 0.7 and row["reaction_time_ms"] < 100:
            return "aimbot"
        if row["survival_rate"] > 0.9 and row["position_changes"] < 20:
            return "wallhack"
        if row["actions_per_minute"] > 190 and row["accuracy"] > 0.6:
            return "triggerbot"
        if row["account_age_days"] < 30 and row["win_rate"] > 70:
            return "smurf"
        return "unknown"
    return "normal"

