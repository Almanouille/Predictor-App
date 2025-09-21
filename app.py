# app.py
import json
import gzip
import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# ---------------- CONFIG ----------------
API_KEY = st.secrets.get("API_KEY", "")
API_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
SEASON = 2025

# ⚠️ FICHIERS À LA RACINE DU REPO
MODEL_PATH = "xgb_model_v2.json.gz"
FEATURES_PATH = "xgb_features_v2.json"
LABELMAP_PATH = "xgb_labelmap_v2.json"

LEAGUES = {
    "Premier League (Angleterre)": 39,
    "Ligue 1 (France)": 61,
    "La Liga (Espagne)": 140,
    "Serie A (Italie)": 135,
    "Bundesliga (Allemagne)": 78,
}

st.set_page_config(page_title="Prédiction Football", layout="centered")
st.title("⚽ Prédiction de match de football")

selected_league = st.selectbox("Choisis une ligue", list(LEAGUES.keys()))
LEAGUE_ID = LEAGUES[selected_league]

# ---------------- UTILS ----------------
def _safe_get(d, *keys, default=0):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d if d is not None else default

@st.cache_resource
def load_artifacts():
    # charge liste d’ordonnancement des features
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_order = json.load(f)

    # charge mapping label -> int (ex: {"away":0,"draw":1,"home":2})
    with open(LABELMAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    # charge Booster XGBoost depuis .json.gz
    with gzip.open(MODEL_PATH, "rb") as gz:
        model_json = gz.read().decode("utf-8")
    booster = xgb.Booster()
    booster.load_model(io.StringIO(model_json))  # charge depuis string

    # dictionnaire inverse pour affichage
    inv_label_map = {v: k for k, v in label_map.items()}
    return booster, feature_order, label_map, inv_label_map

booster, FEATURE_ORDER, LABEL_MAP, INV_LABEL_MAP = load_artifacts()

# ---------------- API ----------------
@st.cache_data
def get_upcoming_matches(league_id: int):
    url = f"{API_URL}/fixtures?league={league_id}&season={SEASON}&next=10"
    r = requests.get(url, headers=HEADERS, timeout=20)
    data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
    return data.get("response", [])

@st.cache_data
def get_name_to_id_mapping(league_id: int):
    url = f"{API_URL}/teams?league={league_id}&season={SEASON}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    data = r.json().get("response", [])
    return {t["team"]["name"]: t["team"]["id"] for t in data}

@st.cache_data
def get_team_stats(team_id: int, league_id: int):
    url = f"{API_URL}/teams/statistics?team={team_id}&season={SEASON}&league={league_id}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    return r.json().get("response", {}) or {}

matches_raw = get_upcoming_matches(LEAGUE_ID)
if not matches_raw:
    st.warning("Aucun match à venir pour cette ligue.")
    st.stop()

options = []
for m in matches_raw:
    fixture = m["fixture"]
    teams = m["teams"]
    label = f"{teams['home']['name']} vs {teams['away']['name']} ({fixture['date'][:10]})"
    options.append({
        "label": label,
        "home": teams["home"]["name"],
        "away": teams["away"]["name"],
        "fixture_id": fixture["id"],
    })

selected = st.selectbox("Choisis un match à venir", options, format_func=lambda x: x["label"])
name_to_id = get_name_to_id_mapping(LEAGUE_ID)

# ---------------- FEATURES (doivent correspondre à FEATURE_ORDER) ----------------
def build_features_row(home_name: str, away_name: str, league_id: int) -> pd.DataFrame:
    # encodage simple des équipes via leur ordre dans la liste des IDs
    home_id = name_to_id.get(home_name)
    away_id = name_to_id.get(away_name)
    if home_id is None or away_id is None:
        st.error("Équipe introuvable dans l'API.")
        st.stop()

    team_ids = list(name_to_id.values())
    home_enc = team_ids.index(home_id)
    away_enc = team_ids.index(away_id)

    # stats API
    hstats = get_team_stats(home_id, league_id)
    astats = get_team_stats(away_id, league_id)

    home_form = str(hstats.get("form", "")).count("W")
    away_form = str(astats.get("form", "")).count("W")

    home_conceded = float(_safe_get(hstats, "goals", "against", "average", "home", default=0) or 0)
    away_conceded = float(_safe_get(astats, "goals", "against", "average", "away", default=0) or 0)

    home_avg_goals = float(_safe_get(hstats, "goals", "for", "average", "home", default=0) or 0)
    away_avg_goals = float(_safe_get(astats, "goals", "for", "average", "away", default=0) or 0)
    goal_diff = home_avg_goals - away_avg_goals

    base = {
        "home_team_enc": home_enc,
        "away_team_enc": away_enc,
        "goal_diff": goal_diff,
        "home_advantage": 1,
        "home_form": home_form,
        "away_form": away_form,
        "home_conceded": home_conceded,
        "away_conceded": away_conceded,
    }

    # aligne exactement les colonnes demandées par le modèle
    row = {col: base.get(col, 0) for col in FEATURE_ORDER}
    return pd.DataFrame([row], columns=FEATURE_ORDER)

# ---------------- AFFICHAGE ----------------
st.markdown("### Détails du match")
st.write(f"- 🏠 Domicile : **{selected['home']}**")
st.write(f"- ✈️ Extérieur : **{selected['away']}**")
st.write(f"- 🏆 Ligue : **{selected_league}**")
st.write(f"- 📅 Date : **{selected['label'].split('(')[-1].replace(')', '')}**")

# ---------------- PREDICTION ----------------
if st.button("Prédire le résultat"):
    X = build_features_row(selected["home"], selected["away"], LEAGUE_ID)
    st.markdown("### Données utilisées pour la prédiction")
    st.dataframe(X)

    try:
        dtest = xgb.DMatrix(X[FEATURE_ORDER].values, feature_names=FEATURE_ORDER)
        proba = booster.predict(dtest)[0]  # shape (3,)
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])

        # mapping classe lisible
        label_name = INV_LABEL_MAP.get(pred_idx, str(pred_idx))
        label_human = {"away": "Victoire extérieure", "draw": "Match nul", "home": "Victoire à domicile"}.get(label_name, label_name)

        st.markdown("### Probabilités")
        st.write({
            "Victoire extérieure (away)": round(float(proba[LABEL_MAP["away"]]), 3),
            "Match nul (draw)": round(float(proba[LABEL_MAP["draw"]]), 3),
            "Victoire à domicile (home)": round(float(proba[LABEL_MAP["home"]]), 3),
        })

        if confidence >= 0.65:
            st.success(f"✅ Prédiction confiante : **{label_human}** (confiance : {confidence:.2%})")
        else:
            st.warning(f"⚠️ Prédiction incertaine : **{label_human}** (confiance : {confidence:.2%})")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
