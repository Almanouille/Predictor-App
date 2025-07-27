import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env only in local/dev
if os.environ.get("ENV") != "production":
    load_dotenv()

API_KEY = os.getenv("API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
LEAGUE_IDS = [
  39,  # Premier League
  140, # La Liga
  135, # Serie A
  78,  # Bundesliga
  61,  # Ligue 1
  88,  # Eredivisie
  94,  # Liga Portugal
  253, # MLS
  71,  # Brasileirão
  2,   # UEFA Champions League
  3,    # UEFA Europa League
]
SEASONS = [2020, 2021, 2022, 2023, 2024]
BASE_URL = "https://v3.football.api-sports.io"
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
RATE_LIMIT_DELAY = 0.1  # 100ms between requests to respect API limits
STORE = ['local'] # todo: Add remote (Remote database storage using Neon PostgreSQL)
NEON_DATABASE_URL = None # todo: Add remote (Remote database storage using Neon PostgreSQL)

