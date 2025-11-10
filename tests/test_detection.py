from pathlib import Path

import pandas as pd
import pytest

from analysis.behavioral_clustering import cluster_play_styles
from analysis.cheat_detection import calculate_cheat_probability, detect_aimbot
from analysis.statistical_analysis import aggregate_player_matches, flag_impossible_performance
from models.train_detector import main as train_main


DATA_PATH = Path("data") / "synthetic_overwatch_matches.csv"
MODEL_PATH = Path("models") / "anomaly_model.pkl"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def ensure_model() -> None:
    if not MODEL_PATH.exists():
        train_main()


def test_dataset_shape():
    df = load_dataset()
    assert len(df) > 1000
    assert {"player_id", "match_id", "headshot_rate", "reaction_time_ms"}.issubset(df.columns)


def test_detect_aimbot_thresholds():
    df = load_dataset()
    suspects = detect_aimbot(df)
    if suspects.empty:
        pytest.skip("No aimbot suspects generated in synthetic dataset.")
    assert (suspects["headshot_rate"] > 0.7).all()
    assert (suspects["reaction_time_ms"] < 100).all()


def test_behavioral_clustering_outputs():
    df = load_dataset()
    clusters = cluster_play_styles(df, n_clusters=4)
    assert "cluster" in clusters.assignments.columns
    assert clusters.assignments["cluster"].nunique() == 4


def test_cheat_probability_scoring():
    ensure_model()
    df = load_dataset()
    scores = calculate_cheat_probability(df, model_path=MODEL_PATH)
    assert not scores.empty
    assert scores["cheat_probability"].between(0, 1).all()
    assert {"player_id", "cheat_type"}.issubset(scores.columns)


def test_flag_impossible_performance():
    df = load_dataset()
    summary = aggregate_player_matches(df)
    flagged = flag_impossible_performance(summary)
    assert set(flagged.columns).issuperset({"player_id", "headshot_rate_mean"})

