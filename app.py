import logging

import requests

import streamlit as st
from config import API_KEY
from models.predictor import FootballPredictor

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('models_trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# League metadata with upcoming match dates
LEAGUE_OPTIONS = [
    {"id": 88, "name": "Eredivisie (Netherlands)",      "next_match": "2025-08-08"},
    {"id": 253, "name": "MLS (USA)",                    "next_match": "2025-08-10"},
    {"id": 39,  "name": "Premier League (England)",     "next_match": "2025-08-16"},
    {"id": 140, "name": "La Liga (Spain)",              "next_match": "2025-08-17"},
    {"id": 61,  "name": "Ligue 1 (France)",             "next_match": "2025-08-17"},
    {"id": 78,  "name": "Bundesliga (Germany)",         "next_match": "2025-08-22"},
    {"id": 135, "name": "Serie A (Italy)",              "next_match": "2025-08-23"},
    {"id": 71,  "name": "Brasileir\u00e3o (Brazil)",         "next_match": "TBD"},
    {"id": 2,   "name": "UEFA Champions League",        "next_match": "2025-09-16"},
    {"id": 3,   "name": "UEFA Europa League",           "next_match": "2025-09-18"},
]

st.title("Football Match Result Predictor")

# Display league selection sorted by next match
league_display = [f"{league['name']} — {league['next_match']}" for league in LEAGUE_OPTIONS]
selected_league = st.selectbox("Select a league:", league_display)

# Get league ID and date from selection
selected_league_data = next((league for league in LEAGUE_OPTIONS if league['name'] in selected_league), None)

if selected_league_data is None:
    st.stop()

league_id = selected_league_data['id']
match_date = selected_league_data['next_match']

# Match selection only if date is defined
if match_date == "TBD":
    st.warning("No match date available for this league yet.")
    st.stop()

# Load matches from API
url = "https://v3.football.api-sports.io/fixtures"
headers = {"x-apisports-key": API_KEY}
params = {
    "league": league_id,
    "season": 2025,  # [?]todo: not hard coded
    "date": match_date
}
response = requests.get(url, headers=headers, params=params)
data = response.json()

if "response" not in data or not data["response"]:
    st.error("No matches found for this league on the specified date.")
    st.stop()

# Show matches to select
matches = data["response"]
match_display = [
    f"{m['teams']['home']['name']} vs {m['teams']['away']['name']}"
    for m in matches
]
selected_match = st.selectbox("Select a match:", match_display)

selected = matches[match_display.index(selected_match)]
team_a = selected['teams']['home']['name']
team_b = selected['teams']['away']['name']

if st.button("Predict"):
    with st.spinner("Predicting result..."):
        predictor = FootballPredictor()
        result = predictor.predict(
            league_id=league_id,
            team_a_name=team_a,
            team_b_name=team_b
        )
        logging.info(f"result= {result}, key={result.keys()}")
        outcome = result['prediction']
        if outcome:
            st.success(f"Predicted outcome: {result['prediction']['outcome']}")
            st.info(f"Confidence: {result['prediction']['confidence']:.2f}")
            st.info(f"Confidence level: {result['explanation']['confidence_level']}")

        else:
            st.error(f"Prediction failed : {result['error']}.")
