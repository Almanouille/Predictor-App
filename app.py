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
    url = f"{API_URL}/fixtures?league={league_id}&season={SEASON}&next=20"
    res = requests.get(url, headers=HEADERS)
    return res.json().get('response', [])

matches_raw = get_upcoming_matches(LEAGUE_ID)

if not matches_raw:
    st.warning("Aucun match à venir trouvé pour cette ligue. Essaie une autre ou réessaie plus tard.")
    st.stop()

# Formatage des matchs pour sélection
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

selected = st.selectbox("🌍 Choisis un match à venir", options, format_func=lambda x: x["label"])

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
st.markdown("### 📌 Vérification du mapping des équipes :")
st.json(team_map)

# Préparation des features (version simple, sans cotes)
def prepare_features(home, away):
    home_enc = team_map.get(home)
    away_enc = team_map.get(away)

    if home_enc is None or away_enc is None:
        st.error(f"Nom d'équipe introuvable dans le mapping : {home if home_enc is None else ''} {away if away_enc is None else ''}")
        st.stop()

    return pd.DataFrame([{
        'home_team_enc': home_enc,
        'away_team_enc': away_enc,
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

    st.markdown("### Encodage des équipes:")
    st.json({"home": team_map.get(selected['home']), "away": team_map.get(selected['away'])})

    st.markdown("### Données utilisées pour la prédiction :")
    st.dataframe(X_match)

    try:
        prediction = model.predict(xgb.DMatrix(X_match))
        st.markdown(f"📊 **Shape prediction** : `{prediction.shape}`")
        st.markdown("📊 **Contenu prediction :**")
        st.dataframe(pd.DataFrame(prediction, columns=["0", "1", "2"]))

        # Choix de la classe prédite (0: ext, 1: nul, 2: dom)
        pred_class = int(pd.DataFrame(prediction).values.argmax(axis=1)[0])

        result_map = {0: "Victoire extérieure", 1: "Match nul", 2: "Victoire à domicile"}
        st.success(f"🔢 Prédiction : **{result_map[pred_class]}**")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
