# app.py — XGBoost + features enrichies (fix load gz)

import io
import json
import gzip
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import xgboost as xgb

# ===============================
# 🔧 CONFIG
# ===============================
st.set_page_config(page_title="Prédiction de match de football", layout="centered")

API_KEY = st.secrets.get("API_KEY", "")
API_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
SEASON = 2025

MODEL_PATH = "xgb_model_v2.json.gz"            # booster JSON gzippé
FEATURES_PATH = "xgb_features_v2.json"         # ordre exact des features
LABELMAP_PATH = "xgb_labelmap_v2.json"         # {away:0, draw:1, home:2}

LEAGUES = {
    "Premier League (Angleterre)": 39,
    "Ligue 1 (France)": 61,
    "La Liga (Espagne)": 140,
    "Serie A (Italie)": 135,
    "Bundesliga (Allemagne)": 78,
}

# ===============================
# 🧰 UTILITAIRES
# ===============================
def _safe_get(d, path, default=0):
    cur = d
    try:
        for key in path:
            cur = cur[key]
        return default if cur is None else cur
    except Exception:
        return default

def _rate_limit_sleep():
    time.sleep(0.3)

# ===============================
# 📦 CHARGEMENT ARTEFACTS
# ===============================
@st.cache_resource
def _load_booster(path: str) -> xgb.Booster:
    booster = xgb.Booster()
    if path.endswith(".gz"):
        # 🔧 IMPORTANT: charger en BYTES, pas en StringIO
        with gzip.open(path, "rb") as f:
            raw: bytes = f.read()
        booster.load_model(bytearray(raw))   # <-- fix
    else:
        booster.load_model(path)
    return booster

@st.cache_resource
def load_artifacts():
    booster = _load_booster(MODEL_PATH)
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_order: List[str] = json.load(f)
    with open(LABELMAP_PATH, "r", encoding="utf-8") as f:
        label_map: Dict[str, int] = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    return booster, feature_order, label_map, inv_label_map, MODEL_PATH

booster, FEATURE_ORDER, LABEL_MAP, INV_LABEL_MAP, MODEL_FILE_SHOWN = load_artifacts()

# ===============================
# 🔌 APIS
# ===============================
@st.cache_data(show_spinner=False)
def get_upcoming_matches(league_id: int):
    url = f"{API_URL}/fixtures?league={league_id}&season={SEASON}&next=10"
    res = requests.get(url, headers=HEADERS, timeout=30)
    data = res.json() if res.ok else {}
    return data.get("response", [])

@st.cache_data(show_spinner=False)
def get_name_to_id_mapping(league_id: int) -> Dict[str, int]:
    url = f"{API_URL}/teams?league={league_id}&season={SEASON}"
    res = requests.get(url, headers=HEADERS, timeout=30)
    data = res.json() if res.ok else {}
    teams = data.get("response", [])
    return {t["team"]["name"]: t["team"]["id"] for t in teams}

@st.cache_data(show_spinner=False)
def get_team_statistics(team_id: int, league_id: int):
    url = f"{API_URL}/teams/statistics?team={team_id}&season={SEASON}&league={league_id}"
    res = requests.get(url, headers=HEADERS, timeout=30)
    return res.json().get("response", {}) if res.ok else {}

@st.cache_data(show_spinner=False)
def get_recent_fixtures(team_id: int, count: int = 5):
    url = f"{API_URL}/fixtures?team={team_id}&season={SEASON}&last={count}"
    res = requests.get(url, headers=HEADERS, timeout=30)
    data = res.json() if res.ok else {}
    return data.get("response", [])

@st.cache_data(show_spinner=False)
def get_injuries(team_id: int):
    url = f"{API_URL}/injuries?team={team_id}&season={SEASON}"
    res = requests.get(url, headers=HEADERS, timeout=30)
    data = res.json() if res.ok else {}
    return len(data.get("response", []))

# ===============================
# 🧮 FEATURES – alignées entraînement
# ===============================
def _form_wld_from_fixtures(fixtures, team_id: int):
    w = d = l = 0
    for fx in fixtures:
        winner_home = _safe_get(fx, ["teams", "home", "winner"], False)
        winner_away = _safe_get(fx, ["teams", "away", "winner"], False)
        home_id = _safe_get(fx, ["teams", "home", "id"], None)
        away_id = _safe_get(fx, ["teams", "away", "id"], None)
        if winner_home is True and home_id == team_id:
            w += 1
        elif winner_away is True and away_id == team_id:
            w += 1
        elif winner_home is False and home_id == team_id:
            l += 1
        elif winner_away is False and away_id == team_id:
            l += 1
        else:
            d += 1
    return w, d, l

def _goals_scored_conceded(fixtures, team_id: int):
    scored = conceded = 0
    for fx in fixtures:
        gh = _safe_get(fx, ["goals", "home"], 0)
        ga = _safe_get(fx, ["goals", "away"], 0)
        if _safe_get(fx, ["teams", "home", "id"], None) == team_id:
            scored += gh; conceded += ga
        else:
            scored += ga; conceded += gh
    return scored, conceded

def _season_rate(stats: dict, side: str, field: str):
    if field == "shots_on_target":
        return _safe_get(stats, ["shots", "on", side], 0)
    if field == "possession":
        val = _safe_get(stats, ["ball", "possession", side], 0)
        try: return float(str(val).replace("%", "")) / 100.0
        except Exception: return 0.0
    if field == "corners":
        return _safe_get(stats, ["corners", "total", side], 0)
    return 0

def build_features_for_pair(home_name: str, away_name: str, league_id: int,
                            feature_order: List[str]) -> pd.DataFrame:
    name2id = get_name_to_id_mapping(league_id)
    if home_name not in name2id or away_name not in name2id:
        st.error("Équipe introuvable dans la ligue sélectionnée."); st.stop()
    home_id = name2id[home_name]; away_id = name2id[away_name]

    fx_home = get_recent_fixtures(home_id, 5); _rate_limit_sleep()
    fx_away = get_recent_fixtures(away_id, 5); _rate_limit_sleep()

    h_w, _, _ = _form_wld_from_fixtures(fx_home, home_id)
    a_w, _, _ = _form_wld_from_fixtures(fx_away, away_id)

    h_scored_5, h_conceded_5 = _goals_scored_conceded(fx_home, home_id)
    a_scored_5, a_conceded_5 = _goals_scored_conceded(fx_away, away_id)

    home_absents = get_injuries(home_id); _rate_limit_sleep()
    away_absents = get_injuries(away_id); _rate_limit_sleep()

    home_stats = get_team_statistics(home_id, league_id); _rate_limit_sleep()
    away_stats = get_team_statistics(away_id, league_id)

    def avg_stat(stats, field):
        v = _season_rate(stats, "total", field) or _season_rate(stats, "home", field) or _season_rate(stats, "away", field)
        return v or 0.0

    shots_home_last5 = avg_stat(home_stats, "shots_on_target")
    shots_away_last5 = avg_stat(away_stats, "shots_on_target")
    poss_home_last5  = avg_stat(home_stats, "possession")
    poss_away_last5  = avg_stat(away_stats, "possession")
    corners_home_last5 = avg_stat(home_stats, "corners")
    corners_away_last5 = avg_stat(away_stats, "corners")

    features_base = {
        "home_team_enc": int(list(name2id.values()).index(home_id)),
        "away_team_enc": int(list(name2id.values()).index(away_id)),
        "home_advantage": 1,
        "goal_diff": float(
            _safe_get(home_stats, ["goals", "for", "average", "home"], 0)
            - _safe_get(away_stats, ["goals", "for", "average", "away"], 0)
        ),
        "home_last5_form": float(h_w),
        "away_last5_form": float(a_w),
        "home_last5_goals_scored": float(h_scored_5),
        "away_last5_goals_scored": float(a_scored_5),
        "home_last5_goals_conceded": float(h_conceded_5),
        "away_last5_goals_conceded": float(a_conceded_5),
        "home_absents": float(home_absents),
        "away_absents": float(away_absents),
        "shots_on_target_home_last5": float(shots_home_last5),
        "shots_on_target_away_last5": float(shots_away_last5),
        "possession_home_last5": float(poss_home_last5),
        "possession_away_last5": float(poss_away_last5),
        "corners_home_last5": float(corners_home_last5),
        "corners_away_last5": float(corners_away_last5),
    }

    row = {col: 0.0 for col in feature_order}
    for k, v in features_base.items():
        if k in row: row[k] = v

    X = pd.DataFrame([row], columns=feature_order)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X

# ===============================
# 🖥️ UI
# ===============================
st.title("⚽ Prédiction de match de football")

selected_league = st.selectbox("Choisis une ligue", list(LEAGUES.keys()))
LEAGUE_ID = LEAGUES[selected_league]

matches_raw = get_upcoming_matches(LEAGUE_ID)
if not matches_raw:
    st.warning("Aucun match à venir pour cette ligue."); st.stop()

options = []
for m in matches_raw:
    fixture = m["fixture"]; teams = m["teams"]
    label = f"{teams['home']['name']} vs {teams['away']['name']} ({fixture['date'][:10]})"
    options.append({"label": label, "home": teams["home"]["name"], "away": teams["away"]["name"],
                    "fixture_id": fixture["id"], "date": fixture["date"][:10]})

selected = st.selectbox("Choisis un match à venir", options, format_func=lambda x: x["label"])
st.caption(f"Modèle chargé : {MODEL_FILE_SHOWN}")

st.markdown("### Détails du match")
st.write(f"- 🏠 Domicile : **{selected['home']}**")
st.write(f"- ✈️ Extérieur : **{selected['away']}**")
st.write(f"- 🏆 Ligue : **{selected_league}**")
st.write(f"- 📅 Date : **{selected['date']}**")

if st.button("Prédire le résultat"):
    with st.spinner("Construction des features en cours…"):
        X_match = build_features_for_pair(selected["home"], selected["away"], LEAGUE_ID, FEATURE_ORDER)

    st.markdown("### Données utilisées pour la prédiction :")
    st.dataframe(X_match)

    try:
        dmat = xgb.DMatrix(X_match)
        proba = booster.predict(dmat)[0]  # [p_away, p_draw, p_home]
        pred_class = int(np.argmax(proba))
        result_text = {0: "Victoire extérieure", 1: "Match nul", 2: "Victoire à domicile"}[pred_class]
        confidence = float(proba[pred_class])

        st.markdown("### Probabilités")
        st.json({
            "Victoire extérieure (away)": round(float(proba[0]), 3),
            "Match nul (draw)": round(float(proba[1]), 3),
            "Victoire à domicile (home)": round(float(proba[2]), 3),
        })

        if confidence >= 0.65:
            st.success(f"✅ Prédiction : **{result_text}** (confiance : {confidence:.2%})")
        else:
            st.warning(f"⚠️ Prédiction : **{result_text}** (confiance faible : {confidence:.2%})")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
