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
    t
