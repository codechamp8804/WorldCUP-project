from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np  # type: ignore

from importlib import import_module
from statistics import mean

try:
    kagglehub = import_module("kagglehub")
except Exception:  # pragma: no cover - fallback for environments without kagglehub
    class _KaggleHubStub:
        @staticmethod
        def dataset_download(slug: str) -> str:
            raise RuntimeError(
                "kagglehub is not installed or not available in this environment. "
                "Install kagglehub or provide a path to results.csv instead."
            )

    kagglehub = _KaggleHubStub()
import pandas as pd  # type: ignore


DATASET_SLUG = "martj42/international-football-results-from-1872-to-2017"
REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "country",
    "neutral",
}


@dataclass
class TeamState:
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    recent_points: tuple[int, ...] = ()
    recent_goals_for: tuple[int, ...] = ()
    recent_goals_against: tuple[int, ...] = ()


def download_kaggle_dataset(data_dir: str | Path = "data") -> Path:
    """Download the Kaggle dataset and copy results.csv into data_dir."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    source = dataset_path / "results.csv"
    if not source.exists():
        matches = list(dataset_path.rglob("results.csv"))
        if not matches:
            raise FileNotFoundError("Could not find results.csv in the downloaded Kaggle dataset.")
        source = matches[0]

    target = data_dir / "results.csv"
    target.write_bytes(source.read_bytes())
    return target


def load_results(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def result_label(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "Home Win"
    if home_score < away_score:
        return "Away Win"
    return "Draw"


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _recent_average(values: tuple[int, ...]) -> float:
    return float(np.mean(values)) if values else 0.0


def _state_features(prefix: str, state: TeamState) -> dict[str, float]:
    return {
        f"{prefix}_matches": float(state.matches),
        f"{prefix}_win_rate": _safe_rate(state.wins, state.matches),
        f"{prefix}_draw_rate": _safe_rate(state.draws, state.matches),
        f"{prefix}_loss_rate": _safe_rate(state.losses, state.matches),
        f"{prefix}_goals_for_avg": _safe_rate(state.goals_for, state.matches),
        f"{prefix}_goals_against_avg": _safe_rate(state.goals_against, state.matches),
        f"{prefix}_recent_points_avg": _recent_average(state.recent_points),
        f"{prefix}_recent_goals_for_avg": _recent_average(state.recent_goals_for),
        f"{prefix}_recent_goals_against_avg": _recent_average(state.recent_goals_against),
    }


def _append_recent(values: tuple[int, ...], new_value: int, size: int = 5) -> tuple[int, ...]:
    return (*values, new_value)[-size:]


def _update_state(state: TeamState, goals_for: int, goals_against: int) -> TeamState:
    if goals_for > goals_against:
        points = 3
        wins, draws, losses = 1, 0, 0
    elif goals_for < goals_against:
        points = 0
        wins, draws, losses = 0, 0, 1
    else:
        points = 1
        wins, draws, losses = 0, 1, 0

    return TeamState(
        matches=state.matches + 1,
        wins=state.wins + wins,
        draws=state.draws + draws,
        losses=state.losses + losses,
        goals_for=state.goals_for + goals_for,
        goals_against=state.goals_against + goals_against,
        recent_points=_append_recent(state.recent_points, points),
        recent_goals_for=_append_recent(state.recent_goals_for, goals_for),
        recent_goals_against=_append_recent(state.recent_goals_against, goals_against),
    )


def build_match_features(results: pd.DataFrame) -> pd.DataFrame:
    """Create pre-match features for every historical match."""
    states: dict[str, TeamState] = {}
    rows: list[dict[str, Any]] = []

    for match in results.itertuples(index=False):
        home_state = states.get(match.home_team, TeamState())
        away_state = states.get(match.away_team, TeamState())

        row = {
            "date": match.date,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "tournament": match.tournament,
            "country": match.country,
            "neutral": bool(match.neutral),
            "is_home_country": match.home_team == match.country and not bool(match.neutral),
            "target": result_label(match.home_score, match.away_score),
        }
        row.update(_state_features("home", home_state))
        row.update(_state_features("away", away_state))
        rows.append(row)

        states[match.home_team] = _update_state(home_state, match.home_score, match.away_score)
        states[match.away_team] = _update_state(away_state, match.away_score, match.home_score)

    features = pd.DataFrame(rows)
    features["match_experience_diff"] = features["home_matches"] - features["away_matches"]
    features["win_rate_diff"] = features["home_win_rate"] - features["away_win_rate"]
    features["recent_points_diff"] = features["home_recent_points_avg"] - features["away_recent_points_avg"]
    features["recent_goal_diff"] = (
        features["home_recent_goals_for_avg"]
        - features["home_recent_goals_against_avg"]
        - features["away_recent_goals_for_avg"]
        + features["away_recent_goals_against_avg"]
    )
    return features


def latest_team_states(results: pd.DataFrame) -> dict[str, TeamState]:
    states: dict[str, TeamState] = {}
    for match in results.itertuples(index=False):
        home_state = states.get(match.home_team, TeamState())
        away_state = states.get(match.away_team, TeamState())
        states[match.home_team] = _update_state(home_state, match.home_score, match.away_score)
        states[match.away_team] = _update_state(away_state, match.away_score, match.home_score)
    return states


def make_prediction_row(
    home_team: str,
    away_team: str,
    tournament: str,
    country: str,
    neutral: bool,
    states: dict[str, TeamState],
) -> pd.DataFrame:
    home_state = states.get(home_team, TeamState())
    away_state = states.get(away_team, TeamState())
    row: dict[str, Any] = {
        "home_team": home_team,
        "away_team": away_team,
        "tournament": tournament,
        "country": country,
        "neutral": neutral,
        "is_home_country": home_team == country and not neutral,
    }
    row.update(_state_features("home", home_state))
    row.update(_state_features("away", away_state))
    row["match_experience_diff"] = row["home_matches"] - row["away_matches"]
    row["win_rate_diff"] = row["home_win_rate"] - row["away_win_rate"]
    row["recent_points_diff"] = row["home_recent_points_avg"] - row["away_recent_points_avg"]
    row["recent_goal_diff"] = (
        row["home_recent_goals_for_avg"]
        - row["home_recent_goals_against_avg"]
        - row["away_recent_goals_for_avg"]
        + row["away_recent_goals_against_avg"]
    )
    return pd.DataFrame([row])