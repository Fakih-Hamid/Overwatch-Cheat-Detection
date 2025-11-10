from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS = [
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
]

CHEAT_LABELS = ["normal", "aimbot", "wallhack", "triggerbot", "smurf"]


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_training_views(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        dataset.groupby("player_id")
        .agg({col: "mean" for col in FEATURE_COLUMNS})
        .join(dataset.groupby("player_id")["cheat_label"].agg(lambda x: x.mode().iat[0]))
    )
    summary.reset_index(inplace=True)

    anomaly_frame = summary.copy()
    anomaly_frame["label"] = (anomaly_frame["cheat_label"] != "normal").astype(int)
    return summary, anomaly_frame


def train_models(summary: pd.DataFrame, anomaly_frame: pd.DataFrame, random_state: int = 42) -> Dict[str, object]:
    scaler = MinMaxScaler()
    features = summary[FEATURE_COLUMNS]
    scaled_features = scaler.fit_transform(features)

    isolation_forest = IsolationForest(
        n_estimators=300,
        contamination=0.18,
        random_state=random_state,
    )
    isolation_forest.fit(scaled_features)

    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, summary["cheat_label"], test_size=0.2, random_state=random_state, stratify=summary["cheat_label"]
    )
    classifier = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        class_weight="balanced",
        random_state=random_state,
    )
    classifier.fit(X_train, y_train)

    test_pred = classifier.predict(X_test)
    probas = classifier.predict_proba(X_test)
    auc_scores = {}
    for idx, label in enumerate(classifier.classes_):
        if label not in CHEAT_LABELS or label == "normal":
            continue
        auc_scores[label] = roc_auc_score((y_test == label).astype(int), probas[:, idx])

    iso_predictions = isolation_forest.predict(scaled_features)
    iso_labels = anomaly_frame["label"].to_numpy()
    iso_detection_rate = ((iso_predictions == -1) & (iso_labels == 1)).sum() / max((iso_labels == 1).sum(), 1)
    iso_false_positive = ((iso_predictions == -1) & (iso_labels == 0)).sum() / max((iso_labels == 0).sum(), 1)

    metrics = {
        "classification_report": classification_report(y_test, test_pred, output_dict=True),
        "roc_auc": auc_scores,
        "isolation_forest": {
            "detection_rate": iso_detection_rate,
            "false_positive_rate": iso_false_positive,
        },
    }

    return {
        "isolation_forest": isolation_forest,
        "classifier": classifier,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
    }


def export_artifacts(artifacts: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cheat detection models on synthetic data.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data") / "synthetic_overwatch_matches.csv",
        help="Path to the synthetic dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("models") / "anomaly_model.pkl",
        help="Path to write the isolation forest artifact.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.data_path)
    summary, anomaly_frame = prepare_training_views(dataset)
    artifacts = train_models(summary, anomaly_frame, random_state=args.random_state)
    export_artifacts(artifacts, args.output_path)


if __name__ == "__main__":
    main()

