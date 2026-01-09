# Overwatch Cheat Analysis

Personal project exploring how statistical analysis, behavioral modeling, and lightweight machine learning techniques can reveal cheating patterns within the Overwatch ecosystem, using fully synthetic data for experimentation.

## Project Goals

- Model common cheating behaviours (aimbot, wallhack, triggerbot, smurfing) with realistic match telemetry.
- Provide reusable detection utilities that surface impossible performance, behavioural anomalies, and model-driven risk scores.
- Visualise risk signals for analysts and prototype a Streamlit dashboard suitable for live operations teams.
  
## Repository Layout

```
overwatch-cheat-analysis/
├── analysis/
│   ├── behavioral_clustering.py      # Behavioural clustering and consistency analysis
│   ├── cheat_detection.py            # Heuristic and ML-driven cheat detection utilities
│   └── statistical_analysis.py       # Z-score auditing and correlation studies
├── data/
│   ├── generate_synthetic_data.py    # Synthetic match telemetry generator
│   └── synthetic_overwatch_matches.csv
├── models/
│   ├── anomaly_model.pkl             # Isolation Forest + classifier artefact
│   └── train_detector.py             # Model training entry point
├── visualization/
│   ├── create_dashboards.py          # Streamlit dashboard to explore detections
│   └── plot_patterns.py              # Matplotlib/Seaborn plotting helpers
├── tests/
│   └── test_detection.py             # Pytest coverage for key detection logic
├── Overwatch_Cheat_Analysis.ipynb    # End-to-end analysis notebook
├── README.md
├── requirements.txt
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate          # On Windows
pip install -r requirements.txt
python data/generate_synthetic_data.py
python models/train_detector.py
pytest
```

The generation script creates player-match level telemetry for 1,000+ matches including normal play, aimbots, wallhacks, triggerbots, and smurf accounts. Model training outputs `models/anomaly_model.pkl`, containing an Isolation Forest for anomaly scoring and a Random Forest classifier for cheat taxonomy.

## Analysis Workflow

1. **Data Exploration (Notebook)**  
   - Profile hero usage, distributions, and correlations.  
   - Validate synthetic data ranges for normal play vs. cheats.  
   - Compute rolling win-rate and performance deltas per player.

2. **Statistical Checks (`analysis/statistical_analysis.py`)**  
   - Z-score scanning for metrics such as headshot rate, accuracy, and reaction time.  
   - Flag impossible human performance (e.g., 95% headshot rate with sub-100ms reaction).  
   - Correlate report volume with skill indicators to quantify community signal quality.

3. **Behavioural Modeling (`analysis/behavioral_clustering.py`)**  
   - K-means clustering across win rate, KD, inputs per minute, and survival rate.  
   - Detect players with unreal consistency (triggerbot signatures) and rapid improvement (potential account sharing or injected cheats).  
   - Surface smurf-like profiles via account age and performance gaps.

4. **Machine Learning Scoring (`analysis/cheat_detection.py`)**  
   - Isolation Forest assigns cheat probability per player based on aggregated telemetry.  
   - Random Forest classifier differentiates aimbot, wallhack, triggerbot, and smurf classes.  
   - Combined heuristic ranking blends reports, suspicious flags, and extreme stats for analyst triage.

5. **Visualisation (`visualization/plot_patterns.py`)**  
   - Scatter plots (headshot rate vs. reaction time), density distributions, cluster maps, and ROC curves.  
   - Streamlit dashboard (`visualization/create_dashboards.py`) for ops teams to filter heroes, review suspects, and inspect model outputs in real time.

## Integration Considerations

- **Telemetry Sources**: Map synthetic features to real telemetry (e.g., `position_changes` from movement logs, `ultimate_efficiency` from ability usage stats).  
- **Pipeline Fit**: Models can run as scheduled batch jobs analysing prior sessions, or as near-real-time microservices evaluating live match data and returning risk scores.  
- **False Positive Management**: Combine statistical thresholds with report weightings, enforce cooldown windows before enforcement, and require multiple signals (Isolation Forest + reports + heuristic trigger) before action.  
- **Scalability**: All routines operate on aggregated player statistics. In production, adapt to distributed stores (BigQuery, Snowflake) and push model inference into scalable endpoints using the trained artefacts saved here.  
- **Auditability**: Every detection routine outputs player IDs with supporting metrics so human reviewers can justify bans or escalate for further investigation.


`Overwatch_Cheat_Analysis.ipynb` demonstrates:

- Data ingestion and preprocessing
- Statistical validation of synthetic telemetry
- Behavioural clustering and centroid interpretation
- Isolation Forest training, evaluation, and ROC charts
- Cheat probability scoring with explanations for top suspects


## Limitations & Future Work

- Synthetic data lacks cross-match team context and objective interactions; extend features with team-based comparisons.  
- Triggerbot modelling is simplified via reaction time variance; production data should consider input timing jitter and click-to-kill intervals.  
- Current models focus on per-player aggregates; a sequence model (LSTM/Temporal CNN) could ingest event timelines for higher fidelity detection.  
- Add streaming ingestion examples (Kafka/Kinesis) and automated retraining workflows for live service environments.

## License

MIT License.  Not affiliated with Blizzard Entertainment.

