import streamlit as st
import pandas as pd
import requests
import xgboost as xgb

# ---------------- CONFIG ----------------
API_KEY = st.secrets["API_KEY"] if "API_KEY" in st.secrets else ""
API_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
SEASON = 2025
MODEL_PATH = "modele_foot_xgb-5.json"

LEAGUES = {
    "Premier League (Angleterre)": 39,
    "Ligue 1 (France)": 61,
    "La Liga (Espagne)": 140,
}

# ---------------- APP ----------------
st.set_page_config(page_title="Prédiction Football", layout="centered")
st.title("Prédiction de match de football")

selected_league = st.selectbox("Choisis une ligue", list(LEAGUES.keys()))
LEAGUE_ID = LEAGUES[selected_league]

@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    return model

model = load_model()

@st.cache_data
def get_upcoming_matches(league_id):
    url = f"{API_URL}/fixtures?league={league_id}&season={SEASON}&next=10"
    res = requests.get(url, headers=HEADERS)
    try:
        data = res.json()
    except Exception:
        return []

    if res.status_code != 200 or "response" not in data:
        st.error("Erreur API - aucun match trouvé.")
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
def get_team_mapping(league_id):
    url = f"{API_URL}/teams?league={league_id}&season={SEASON}"
    res = requests.get(url, headers=HEADERS)
    teams = res.json()['response']
    return {team['team']['id']: team['team']['name'] for team in teams}

def get_name_to_id_mapping(league_id):
    url = f"{API_URL}/teams?league={league_id}&season={SEASON}"
    res = requests.get(url, headers=HEADERS)
    teams = res.json()['response']
    return {team['team']['name']: team['team']['id'] for team in teams}

team_map = get_team_mapping(LEAGUE_ID)
name_to_id_map = get_name_to_id_mapping(LEAGUE_ID)

@st.cache_data
def get_team_stats(team_id, league_id):
    url = f"{API_URL}/teams/statistics?team={team_id}&season={SEASON}&league={league_id}"
    res = requests.get(url, headers=HEADERS)
    return res.json().get('response', {})

@st.cache_data
def prepare_features(home, away):
    home_id = name_to_id_map.get(home)
    away_id = name_to_id_map.get(away)

    if home_id is None or away_id is None:
        st.error("Équipe introuvable.")
        st.stop()

    team_ids = list(name_to_id_map.values())
    home_enc = team_ids.index(home_id)
    away_enc = team_ids.index(away_id)

    home_stats = get_team_stats(home_id, LEAGUE_ID)
    away_stats = get_team_stats(away_id, LEAGUE_ID)

    home_form = home_stats.get("form", "").count("W")
    away_form = away_stats.get("form", "").count("W")

    home_conceded = float(home_stats["goals"]["against"]["average"]["home"] or 0)
    away_conceded = float(away_stats["goals"]["against"]["average"]["away"] or 0)

    home_avg_goals = float(home_stats["goals"]["for"]["average"]["home"] or 0)
    away_avg_goals = float(away_stats["goals"]["for"]["average"]["away"] or 0)
    goal_diff = home_avg_goals - away_avg_goals

    return pd.DataFrame([{
        "home_team_enc": home_enc,
        "away_team_enc": away_enc,
        "goal_diff": goal_diff,
        "home_advantage": 1,
        "home_form": home_form,
        "away_form": away_form,
        "home_conceded": home_conceded,
        "away_conceded": away_conceded
    }])

# ---------------- AFFICHAGE DU MATCH ----------------
st.markdown("### Détails du match")
st.write(f"- Équipe à domicile : {selected['home']}")
st.write(f"- Équipe à l'extérieur : {selected['away']}")
st.write(f"- Ligue : {selected_league}")
st.write(f"- Date prévue : {selected['label'].split('(')[-1].replace(')', '')}")

# ---------------- PRÉDICTION ----------------
if st.button("Prédire le résultat"):
    X_match = prepare_features(selected['home'], selected['away'])

    st.markdown("### Données utilisées pour la prédiction :")
    st.dataframe(X_match)

    try:
        proba = model.predict(xgb.DMatrix(X_match))[0]  # tableau de 3 probabilités
        st.markdown("### Probabilités prédites :")
        st.write({
            "Victoire extérieure (0)": round(proba[0], 3),
            "Match nul (1)": round(proba[1], 3),
            "Victoire à domicile (2)": round(proba[2], 3),
        })

        pred_class = int(proba.argmax())
        confidence = proba[pred_class]
        result_map = {0: "Victoire extérieure", 1: "Match nul", 2: "Victoire à domicile"}

        if confidence >= 0.65:
            st.success(f"✅ Prédiction confiante : {result_map[pred_class]} (confiance : {confidence:.2%})")
        else:
            st.warning(f"❌ Pas de prédiction fiable (confiance trop faible : {confidence:.2%})")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
