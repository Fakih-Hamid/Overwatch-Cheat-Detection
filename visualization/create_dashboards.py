from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from analysis.behavioral_clustering import cluster_play_styles, describe_clusters
from analysis.cheat_detection import (
    calculate_cheat_probability,
    detect_aimbot,
    detect_triggerbot,
    detect_wallhack,
    identify_smurf_accounts,
    rank_suspicious_players,
)
from analysis.statistical_analysis import aggregate_player_matches, compute_z_score_outliers, flag_impossible_performance
from visualization.plot_patterns import distribution_plot, scatter_headshot_vs_reaction


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_dashboard(dataset_path: Path, model_path: Path) -> None:
    st.title("Overwatch Cheat Detection Overview")
    st.markdown(
        "<div style='text-align: right; font-weight: 600;'>By Fakih Hamid</div>",
        unsafe_allow_html=True,
    )
    dataset = load_dataset(dataset_path)

    st.sidebar.header("Filters")
    hero_filter = st.sidebar.multiselect("Hero", sorted(dataset["hero"].unique()))
    label_filter = st.sidebar.multiselect("Label", sorted(dataset["cheat_label"].unique()))

    filtered = dataset.copy()
    if hero_filter:
        filtered = filtered[filtered["hero"].isin(hero_filter)]
    if label_filter:
        filtered = filtered[filtered["cheat_label"].isin(label_filter)]

    st.subheader("Dataset Snapshot")
    st.dataframe(filtered.head(50))

    st.subheader("Feature Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig = scatter_headshot_vs_reaction(filtered)
        st.pyplot(fig)
    with col2:
        fig = distribution_plot(filtered, "kd_ratio")
        st.pyplot(fig)

    st.subheader("Statistical Monitoring")
    player_summary = aggregate_player_matches(filtered)
    outliers = compute_z_score_outliers(
        player_summary,
        metrics=["headshot_rate_mean", "accuracy_mean", "kd_ratio_mean", "win_rate_mean"],
    )
    st.write("Z-Score Outliers", outliers)
    st.write("Impossible Stats", flag_impossible_performance(player_summary))

    st.subheader("Behavioral Clustering")
    cluster_result = cluster_play_styles(filtered)
    st.dataframe(describe_clusters(cluster_result))

    st.subheader("Detections")
    st.write("Aimbot Suspects", detect_aimbot(filtered))
    st.write("Wallhack Suspects", detect_wallhack(filtered))
    st.write("Triggerbot Suspects", detect_triggerbot(filtered))
    st.write("Potential Smurfs", identify_smurf_accounts(filtered))
    st.write("Risk Ranking", rank_suspicious_players(filtered).head(20))

    st.subheader("Model Scoring")
    scoring = calculate_cheat_probability(filtered, model_path=model_path)
    st.dataframe(scoring.sort_values("cheat_probability", ascending=False).head(25))


def main() -> None:
    default_data = Path("data") / "synthetic_overwatch_matches.csv"
    default_model = Path("models") / "anomaly_model.pkl"
    build_dashboard(default_data, default_model)


if __name__ == "__main__":
    main()

