import streamlit as st
import pandas as pd
import requests
import xgboost as xgb

# ---------------------- CONFIG ----------------------
API_KEY = st.secrets["API_KEY"] if "API_KEY" in st.secrets else ""
API_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
SEASON = 2024
MODEL_PATH = "modele_foot_xgb.json"

# Dictionnaire des ligues
LEAGUES = {
    "Premier League (Angleterre)": 39,
    "Ligue 1 (France)": 61,
    "La Liga (Espagne)": 140,
    "Serie A (Italie)": 135,
    "Bundesliga (Allemagne)": 78
}

# ---------------------- APP ----------------------
st.set_page_config(page_title="Prédiction Football", layout="centered")
st.title("🏀 Prédiction de match de football")

# Choix de la ligue
selected_league = st.selectbox("🌟 Choisis une ligue", list(LEAGUES.keys()))
LEAGUE_ID = LEAGUES[selected_league]

# Chargement du modèle
@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    return model

model = load_model()

# Récupération des matchs à venir
@st.cache_data
def get_upcoming_matches(league_id):
    url = f"{API_URL}/fixtures?league={league_id}&season={SEASON}&next=10"
    res = requests.get(url, headers=HEADERS)
    matches = res.json()['response']
    options = []
    for m in matches:
        fixture = m['fixture']
        teams = m['teams']
        label = f"{teams['home']['name']} vs {teams['away']['name']} ({fixture['date'][:10]})"
        options.append({
            "label": label,
            "home": teams['home']['name'],
            "away": teams['away']['name'],
            "fixture_id": fixture['id']
        })
    return options

matches = get_upcoming_matches(LEAGUE_ID)

selected = st.selectbox("🌍 Choisis un match à venir", matches, format_func=lambda x: x["label"])

# Dummy encoder - remplacer par un vrai encodage ou mappage
@st.cache_data
def get_team_mapping(league_id):
    teams = []
    url = f"{API_URL}/teams?league={league_id}&season={SEASON}"
    res = requests.get(url, headers=HEADERS)
    for t in res.json()['response']:
        teams.append(t['team']['name'])
    return {name: idx for idx, name in enumerate(sorted(teams))}

team_map = get_team_mapping(LEAGUE_ID)

# Préparation des features (version simple, sans cotes)
def prepare_features(home, away):
    return pd.DataFrame([{
        'home_team_enc': team_map.get(home, 0),
        'away_team_enc': team_map.get(away, 0),
        'goal_diff': 0,
        'home_advantage': 1
    }])

# Afficher des stats fictives (bonus visuel)
def display_match_info(match):
    st.markdown("### 📋 Détails du match")
    st.write(f"- **Équipe à domicile** : {match['home']}")
    st.write(f"- **Équipe à l'extérieur** : {match['away']}")
    st.write(f"- **Ligue** : {selected_league}")
    st.write(f"- **Date prévue** : {match['label'].split('(')[-1].replace(')', '')}")

# Affichage des infos du match sélectionné
display_match_info(selected)

if st.button("🔢 Prédire le résultat"):
    X_match = prepare_features(selected['home'], selected['away'])
    pred = int(model.predict(xgb.DMatrix(X_match))[0])
    result_map = {0: "Victoire extérieure", 1: "Match nul", 2: "Victoire à domicile"}
    st.success(f"🔢 Prédiction : **{result_map[pred]}**")
