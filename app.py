from __future__ import annotations

import joblib  # type: ignore
from pathlib import Path

import pandas as pd  # type: ignore[import]
import streamlit as st  # type: ignore[import]

from football_features import latest_team_states, load_results, make_prediction_row


MODEL_PATH = Path("artifacts/model.joblib")


@st.cache_resource
def load_artifacts() -> dict:
    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python train_model.py --data data/results.csv` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_historical_results(results_path: str) -> pd.DataFrame:
    return load_results(results_path)


st.set_page_config(page_title="World Cup Match Predictor", page_icon="⚽", layout="centered")
st.set_page_config(page_title="World Cup Match Predictor", layout="centered")
st.title("World Cup Match Outcome Predictor")
st.caption("Predicts Home Win, Draw, or Away Win from historical international football results.")

artifacts = load_artifacts()
model = artifacts["model"]
teams = artifacts["teams"]
tournaments = artifacts["tournaments"]
countries = artifacts["countries"]
results = load_historical_results(artifacts["results_path"])
states = latest_team_states(results)

with st.sidebar:
    st.header("Match setup")
    default_tournament = "FIFA World Cup" if "FIFA World Cup" in tournaments else tournaments[0]
    tournament = st.selectbox("Tournament", tournaments, index=tournaments.index(default_tournament))
    country = st.selectbox("Host country", countries, index=0)
    neutral = st.checkbox("Neutral venue", value=False)

home_default = teams.index("Brazil") if "Brazil" in teams else 0
away_default = teams.index("Argentina") if "Argentina" in teams else min(1, len(teams) - 1)

home_team = st.selectbox("Home team", teams, index=home_default)
away_team = st.selectbox("Away team", teams, index=away_default)

if home_team == away_team:
    st.warning("Choose two different teams.")
    st.stop()

prediction_row = make_prediction_row(
    home_team=home_team,
    away_team=away_team,
    tournament=tournament,
    country=country,
    neutral=neutral,
    states=states,
)

prediction = model.predict(prediction_row)[0]
probabilities = model.predict_proba(prediction_row)[0]
probability_table = (
    pd.DataFrame({"Outcome": model.classes_, "Probability": probabilities})
    .sort_values("Probability", ascending=False)
    .reset_index(drop=True)
)

st.subheader(f"{home_team} vs {away_team}")
st.metric("Predicted outcome", prediction)

st.progress(float(probability_table.loc[0, "Probability"]))
st.dataframe(
    probability_table.assign(Probability=lambda df: (df["Probability"] * 100).round(1).astype(str) + "%"),
    hide_index=True,
    use_container_width=True,
)

st.subheader("Feature snapshot")
snapshot = prediction_row[
    [
        "home_matches",
        "away_matches",
        "home_win_rate",
        "away_win_rate",
        "home_recent_points_avg",
        "away_recent_points_avg",
        "recent_points_diff",
        "recent_goal_diff",
        "neutral",
        "is_home_country",
    ]
].T.rename(columns={0: "Value"})
st.dataframe(snapshot, use_container_width=True)

st.caption(
    "This is a portfolio ML model, not a betting tool. Football has high randomness, "
    "and historical results do not capture injuries, squads, tactics, or current rankings."
)