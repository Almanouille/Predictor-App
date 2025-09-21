import streamlit as st
import pandas as pd
import numpy as np
import requests
import json, gzip
import xgboost as xgb

# ---------------- CONFIG ----------------
API_KEY = st.secrets["API_KEY"] if "API_KEY" in st.secrets else ""
API_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
SEASON = 2025

# Modèle XGBoost compressé + méta
MODEL_PATH = "models/xgb_model_v2.json.gz"
FEATURES_JSON = "models/xgb_features_v2.json"
LABELMAP_JSON = "models/xgb_labelmap_v2.json"

LEAGUES = {
    "Premier League (Angleterre)": 39,
    "Ligue 1 (France)": 61,
    "La Liga (Espagne)": 140,
    "Serie A (Italie)": 135,
    "Bundesliga (Allemagne)": 78,
}

# ---------------- APP ----------------
st.set_page_config(page_title="Prédiction Football", layout="centered")
st.title("⚽ Prédiction de match de football")

selected_league = st.selectbox("Choisis une ligue", list(LEAGUES.keys()))
LEAGUE_ID = LEAGUES[selected_league]

# ---------------- MODEL ----------------
@st.cache_resource
def load_xgb():
    # Décompresse en mémoire et charge le Booster
    with gzip.open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()
    bst = xgb.Booster()
    bst.load_model(bytearray(model_bytes))

    # Charge la liste des features et la table des labels
    with open(FEATURES_JSON) as f:
        feature_names = json.load(f)["feature_names"]
    with open(LABELMAP_JSON) as f:
        label_map = json.load(f)  # ex: {"away":0,"draw":1,"home":2}
    inv_label = {v: k for k, v in label_map.items()}
    return bst, feature_names, inv_label

xgb_model, TRAIN_FEATURES, INV_LABEL = load_xgb()

# ---------------- API FUNCTIONS ----------------
@st.cache_data
def get_upcoming_matches(league_id):
    url = f"{API_URL}/fixtures?league={league_id}&season={SEASON}&next=10"
    res = requests.get(url, headers=HEADERS)
    try:
        data = res.json()
    except Exception:
        return []
    if res.status_code != 200 or "response" not in data:
        return []
    return data["response"]

matches_raw = get_upcoming_matches(LEAGUE_ID)

if not matches_raw:
    st.warning("Aucun match à venir pour cette ligue.")
    st.stop()

options = []
for m in matches_raw:
    fixture = m['fixture']
    teams = m['teams']
    label = f"{teams['home']['name']} vs {teams['away']['name']} ({fixture['date'][:10]})"
    options.append({
        "label": label,
        "home": teams['home']['name'],
        "away": teams['away']['name'],
        "fixture_id": fixture['id']
    })

selected = st.selectbox("Choisis un match à venir", options, format_func=lambda x: x["label"])

@st.cache_data
def get_name_to_id_mapping(league_id):
    url = f"{API_URL}/teams?league={league_id}&season={SEASON}"
    res = requests.get(url, headers=HEADERS)
    teams = res.json().get('response', [])
    return {team['team']['name']: team['team']['id'] for team in teams}

name_to_id_map = get_name_to_id_mapping(LEAGUE_ID)

@st.cache_data
def get_team_stats(team_id, league_id):
    url = f"{API_URL}/teams/statistics?team={team_id}&season={SEASON}&league={league_id}"
    res = requests.get(url, headers=HEADERS)
    return res.json().get('response', {})

# ---------------- FEATURES ----------------
@st.cache_data
def prepare_features(home, away):
    # IDs
    home_id = name_to_id_map.get(home)
    away_id = name_to_id_map.get(away)
    if home_id is None or away_id is None:
        st.error("Équipe introuvable.")
        st.stop()

    # Encodage basique d'équipes (position dans la liste)
    team_ids = list(name_to_id_map.values())
    home_enc = team_ids.index(home_id)
    away_enc = team_ids.index(away_id)

    # Stats équipe (API-Football)
    home_stats = get_team_stats(home_id, LEAGUE_ID)
    away_stats = get_team_stats(away_id, LEAGUE_ID)

    home_form = home_stats.get("form", "").count("W")
    away_form = away_stats.get("form", "").count("W")

    home_conceded = float(home_stats.get("goals", {}).get("against", {}).get("average", {}).get("home", 0) or 0)
    away_conceded = float(away_stats.get("goals", {}).get("against", {}).get("average", {}).get("away", 0) or 0)

    home_avg_goals = float(home_stats.get("goals", {}).get("for", {}).get("average", {}).get("home", 0) or 0)
    away_avg_goals = float(away_stats.get("goals", {}).get("for", {}).get("average", {}).get("away", 0) or 0)
    goal_diff = home_avg_goals - away_avg_goals

    # DataFrame initial conforme aux features de training (on complètera ensuite)
    df_feats = pd.DataFrame([{
        "home_team_enc": home_enc,
        "away_team_enc": away_enc,
        "goal_diff": goal_diff,
        "home_advantage": 1,
        "home_form": home_form,
        "away_form": away_form,
        "home_conceded": home_conceded,
        "away_conceded": away_conceded
    }])

    # Réindexer sur l'ordre exact des features du modèle et remplir les manquants à 0
    df_feats = df_feats.reindex(columns=TRAIN_FEATURES, fill_value=0)

    # Types numériques pour XGBoost
    for c in df_feats.columns:
        if df_feats[c].dtype == "object":
            df_feats[c] = pd.to_numeric(df_feats[c], errors="coerce").fillna(0)

    return df_feats

# ---------------- AFFICHAGE ----------------
st.markdown("### Détails du match")
st.write(f"- 🏠 Domicile : {selected['home']}")
st.write(f"- ✈️ Extérieur : {selected['away']}")
st.write(f"- 🏆 Ligue : {selected_league}")
st.write(f"- 📅 Date : {selected['label'].split('(')[-1].replace(')', '')}")

# ---------------- PREDICTION ----------------
if st.button("Prédire le résultat"):
    X_match = prepare_features(selected['home'], selected['away'])

    st.markdown("### Données utilisées pour la prédiction :")
    st.dataframe(X_match)

    try:
        # DMatrix avec noms de colonnes alignés au training
        dmat = xgb.DMatrix(X_match, feature_names=TRAIN_FEATURES)

        proba = xgb_model.predict(dmat)[0]  # [p_away, p_draw, p_home]
        pred_class = int(np.argmax(proba))

        st.markdown("### Probabilités prédites :")
        st.write({
            "Victoire extérieure (away=0)": round(float(proba[0]), 3),
            "Match nul (draw=1)": round(float(proba[1]), 3),
            "Victoire à domicile (home=2)": round(float(proba[2]), 3),
        })

        result_map = {0: "Victoire extérieure", 1: "Match nul", 2: "Victoire à domicile"}
        confidence = float(proba[pred_class])

        if confidence >= 0.65:
            st.success(f"✅ Prédiction confiante : {result_map[pred_class]} (confiance : {confidence:.2%})")
        else:
            st.warning(f"ℹ️ Prédiction : {result_map[pred_class]} (confiance : {confidence:.2%})")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
