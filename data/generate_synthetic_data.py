import argparse
import pathlib
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


HERO_POOL: Tuple[str, ...] = (
    "Tracer",
    "Widowmaker",
    "Soldier: 76",
    "Hanzo",
    "Ashe",
    "Sojourn",
    "McCree",
    "Ana",
    "Mercy",
    "Moira",
    "Genji",
    "Sombra",
    "Zenyatta",
    "Sigma",
    "Winston",
    "Reinhardt",
)

CHEAT_TYPES: Tuple[str, ...] = ("aimbot", "wallhack", "triggerbot", "smurf")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_players(num_normal: int, num_cheaters: int) -> pd.DataFrame:
    normal_ids = [f"N{idx:04d}" for idx in range(num_normal)]
    cheat_ids = [f"C{idx:04d}" for idx in range(num_cheaters)]

    normals = pd.DataFrame(
        {
            "player_id": normal_ids,
            "cheat_type": "normal",
            "account_age_days": np.random.randint(90, 1800, size=num_normal),
            "baseline_skill": np.random.normal(loc=0.0, scale=1.0, size=num_normal),
        }
    )

    cheat_categories = np.random.choice(CHEAT_TYPES, size=num_cheaters, p=[0.35, 0.25, 0.2, 0.2])
    cheaters = pd.DataFrame(
        {
            "player_id": cheat_ids,
            "cheat_type": cheat_categories,
            "account_age_days": np.where(
                cheat_categories == "smurf",
                np.random.randint(1, 45, size=num_cheaters),
                np.random.randint(60, 720, size=num_cheaters),
            ),
            "baseline_skill": np.random.normal(loc=1.5, scale=0.7, size=num_cheaters),
        }
    )

    return pd.concat([normals, cheaters], ignore_index=True)


def simulate_match_counts(players: pd.DataFrame) -> np.ndarray:
    base_matches = np.random.randint(6, 18, size=len(players))
    adjustments = np.zeros(len(players), dtype=int)
    for idx, cheat_type in enumerate(players["cheat_type"]):
        if cheat_type == "normal":
            adjustments[idx] = np.random.randint(-2, 3)
        elif cheat_type == "aimbot":
            adjustments[idx] = np.random.randint(3, 8)
        elif cheat_type == "wallhack":
            adjustments[idx] = np.random.randint(0, 6)
        elif cheat_type == "triggerbot":
            adjustments[idx] = np.random.randint(2, 7)
        elif cheat_type == "smurf":
            adjustments[idx] = np.random.randint(5, 12)
    return np.clip(base_matches + adjustments, 3, None)


def create_match_records(players: pd.DataFrame) -> pd.DataFrame:
    match_rows: List[Dict[str, object]] = []
    match_counter = 1

    matches_per_player = simulate_match_counts(players)
    for player, player_row in players.iterrows():
        num_matches = int(matches_per_player[player])
        for _ in range(num_matches):
            hero = random.choice(HERO_POOL)
            base_stats = base_stat_profile(hero, player_row["baseline_skill"])
            adjusted_stats = apply_cheat_profile(base_stats, player_row)

            match_rows.append(
                {
                    "player_id": player_row["player_id"],
                    "match_id": f"M{match_counter:05d}",
                    "hero": hero,
                    **adjusted_stats,
                    "account_age_days": max(
                        player_row["account_age_days"], adjusted_stats.get("account_age_days", 0)
                    ),
                    "win_loss": np.random.choice(["win", "loss"], p=win_probability(adjusted_stats)),
                    "report_count": adjusted_stats["report_count"],
                    "suspicious_flags": adjusted_stats["suspicious_flags"],
                    "cheat_label": player_row["cheat_type"],
                }
            )
            match_counter += 1

    return pd.DataFrame(match_rows)


def base_stat_profile(hero: str, skill_factor: float) -> Dict[str, float]:
    headshot_rate = np.clip(np.random.normal(loc=0.3 + 0.03 * skill_factor, scale=0.05), 0.1, 0.75)
    accuracy = np.clip(np.random.normal(loc=0.45 + 0.04 * skill_factor, scale=0.08), 0.15, 0.9)
    kills = np.random.normal(loc=18 + 3 * skill_factor, scale=5)
    deaths = np.random.normal(loc=11 - 1.5 * skill_factor, scale=3)
    deaths = max(deaths, 0.5)
    kd_ratio = kills / max(deaths, 1)
    damage_done = np.random.normal(loc=8000 + 1500 * skill_factor, scale=1800)
    healing_done = np.random.normal(loc=2000 + 1200 * skill_factor, scale=1000)
    objective_time = np.clip(np.random.normal(loc=60 + 10 * skill_factor, scale=20), 5, 240)
    reaction_time = np.clip(np.random.normal(loc=220 - 20 * skill_factor, scale=35), 120, 400)
    apm = np.random.normal(loc=170 + 10 * skill_factor, scale=25)
    position_changes = np.random.normal(loc=42 + 4 * skill_factor, scale=10)
    survival_rate = np.clip(1 - (deaths / 30), 0.2, 0.98)
    ultimate_efficiency = np.clip(np.random.normal(loc=0.65 + 0.05 * skill_factor, scale=0.08), 0.2, 1.0)
    report_count = max(int(np.random.poisson(lam=0.3 + max(skill_factor, 0) * 0.1)), 0)

    base = {
        "headshot_rate": headshot_rate,
        "accuracy": accuracy,
        "kills": max(kills, 0),
        "deaths": deaths,
        "damage_done": max(damage_done, 0),
        "healing_done": max(healing_done, 0),
        "objective_time": objective_time,
        "reaction_time_ms": reaction_time,
        "actions_per_minute": max(apm, 40),
        "position_changes": max(position_changes, 5),
        "survival_rate": survival_rate,
        "ultimate_efficiency": ultimate_efficiency,
        "account_age_days": 0,
        "win_loss": "loss",
        "kd_ratio": kd_ratio,
        "report_count": report_count,
        "suspicious_flags": 0,
    }

    if hero in {"Mercy", "Ana", "Moira"}:
        base["healing_done"] = max(base["healing_done"] * 1.6, 3000)
    if hero in {"Reinhardt", "Winston"}:
        base["damage_done"] *= 0.8

    return base


def apply_cheat_profile(stats: Dict[str, float], player_row: pd.Series) -> Dict[str, float]:
    cheat_type = player_row["cheat_type"]
    modified = stats.copy()

    if cheat_type == "normal":
        return apply_normal_constraints(modified)

    modified["suspicious_flags"] = 1
    modified["report_count"] = max(int(np.random.poisson(lam=3.5)), stats["report_count"])

    if cheat_type == "aimbot":
        modified["headshot_rate"] = np.random.uniform(0.7, 0.95)
        modified["accuracy"] = np.random.uniform(0.7, 0.92)
        modified["reaction_time_ms"] = np.random.uniform(55, 95)
        modified["kills"] = max(np.random.normal(loc=28, scale=4), modified["kills"])
        modified["kd_ratio"] = modified["kills"] / max(modified["deaths"], 1)
        modified["actions_per_minute"] = max(modified["actions_per_minute"], np.random.uniform(180, 230))

    elif cheat_type == "wallhack":
        modified["survival_rate"] = np.random.uniform(0.88, 0.99)
        modified["position_changes"] = np.random.uniform(15, 30)
        modified["reaction_time_ms"] = np.random.uniform(80, 140)
        modified["accuracy"] = max(modified["accuracy"], np.random.uniform(0.6, 0.8))
        modified["damage_done"] = max(modified["damage_done"], np.random.normal(loc=11000, scale=1500))

    elif cheat_type == "triggerbot":
        modified["headshot_rate"] = np.random.uniform(0.55, 0.8)
        modified["accuracy"] = np.random.uniform(0.65, 0.85)
        modified["reaction_time_ms"] = np.random.uniform(60, 110)
        modified["actions_per_minute"] = np.random.uniform(150, 210)

    elif cheat_type == "smurf":
        modified["headshot_rate"] = np.random.uniform(0.45, 0.7)
        modified["accuracy"] = np.random.uniform(0.6, 0.85)
        modified["kills"] = max(modified["kills"], np.random.normal(loc=24, scale=5))
        modified["deaths"] = max(np.random.normal(loc=7, scale=2), 1)
        modified["kd_ratio"] = modified["kills"] / max(modified["deaths"], 1)
        modified["account_age_days"] = max(player_row["account_age_days"], 1)

    modified["win_loss"] = np.random.choice(["win", "loss"], p=win_probability(modified))
    modified["suspicious_flags"] += int(modified["headshot_rate"] > 0.7)
    modified["suspicious_flags"] += int(modified["reaction_time_ms"] < 100)
    modified["suspicious_flags"] += int(modified["survival_rate"] > 0.9)

    return apply_normal_constraints(modified)


def apply_normal_constraints(stats: Dict[str, float]) -> Dict[str, float]:
    stats["headshot_rate"] = float(np.clip(stats["headshot_rate"], 0.05, 1.0))
    stats["accuracy"] = float(np.clip(stats["accuracy"], 0.1, 1.0))
    stats["kills"] = float(max(stats["kills"], 0))
    stats["deaths"] = float(max(stats["deaths"], 0.1))
    stats["kd_ratio"] = float(stats["kills"] / max(stats["deaths"], 1))
    stats["damage_done"] = float(max(stats["damage_done"], 0))
    stats["healing_done"] = float(max(stats["healing_done"], 0))
    stats["objective_time"] = float(max(stats["objective_time"], 0))
    stats["reaction_time_ms"] = float(max(stats["reaction_time_ms"], 30))
    stats["actions_per_minute"] = float(max(stats["actions_per_minute"], 40))
    stats["position_changes"] = float(max(stats["position_changes"], 1))
    stats["survival_rate"] = float(np.clip(stats["survival_rate"], 0, 1))
    stats["ultimate_efficiency"] = float(np.clip(stats["ultimate_efficiency"], 0, 1))
    stats["report_count"] = int(max(stats["report_count"], 0))
    stats["suspicious_flags"] = int(max(stats["suspicious_flags"], 0))
    return stats


def win_probability(stats: Dict[str, float]) -> Tuple[float, float]:
    offensive = stats["kd_ratio"] + stats["accuracy"] + stats["headshot_rate"]
    support = stats["ultimate_efficiency"] + stats["survival_rate"]
    score = offensive * 0.5 + support * 0.5
    win_prob = np.clip(0.35 + 0.1 * score, 0.2, 0.85)
    return win_prob, 1 - win_prob


def generate_dataset(num_normal: int, num_cheaters: int) -> pd.DataFrame:
    players = generate_players(num_normal, num_cheaters)
    dataset = create_match_records(players)

    # Align metrics with requested ranges for base behaviour
    normals = dataset["cheat_label"] == "normal"
    dataset.loc[normals, "headshot_rate"] = dataset.loc[normals, "headshot_rate"].clip(0.15, 0.45)
    dataset.loc[normals, "reaction_time_ms"] = dataset.loc[normals, "reaction_time_ms"].clip(150, 300)
    dataset.loc[normals, "kd_ratio"] = dataset.loc[normals, "kd_ratio"].clip(0.8, 2.5)
    dataset.loc[normals, "accuracy"] = dataset.loc[normals, "accuracy"].clip(0.3, 0.6)
    dataset.loc[normals, "survival_rate"] = dataset.loc[normals, "survival_rate"].clip(0.2, 0.85)
    dataset.loc[normals, "report_count"] = dataset.loc[normals, "report_count"].clip(0, 3)

    dataset["win_indicator"] = (dataset["win_loss"] == "win").astype(int)
    rolling = dataset.groupby("player_id")["win_indicator"].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    dataset["win_rate"] = (rolling * 100).round(2)
    dataset.drop(columns=["win_indicator"], inplace=True)

    return dataset


def save_dataset(dataset: pd.DataFrame, output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic Overwatch match dataset.")
    parser.add_argument("--normal", type=int, default=800, help="Number of normal players.")
    parser.add_argument("--cheaters", type=int, default=200, help="Number of cheating players.")
    parser.add_argument("--seed", type=int, default=22, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("data") / "synthetic_overwatch_matches.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    dataset = generate_dataset(args.normal, args.cheaters)
    save_dataset(dataset, args.output)


if __name__ == "__main__":
    main()

