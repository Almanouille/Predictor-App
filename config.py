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
  140,  # La Liga
  135,  # Serie A
  78,  # Bundesliga
  61,  # Ligue 1
  88,  # Eredivisie
  253, # MLS
  71,  # Brasileirão
  2,  # UEFA Champions League
  3,  # UEFA Europa League
]
SEASONS = [2020, 2021, 2022, 2023, 2024]
BASE_URL = "https://v3.football.api-sports.io"
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"

RATE_LIMIT_DELAY = 0.1  # 100ms between requests to respect API limits
STORE = ['local'] # todo: Add remote (Remote database storage using Neon PostgreSQL)
NEON_DATABASE_URL = None # todo: Add remote (Remote database storage using Neon PostgreSQL)
FEATURE_NAMES = [
    'is_top_league', 'league_id', 'season',
    'team_a_goals_against_total', 'team_a_goals_for_total', 'team_a_goal_difference_avg',
    'team_a_goals_against_avg', 'team_a_goals_for_avg', 'team_a_league_position',
    'team_a_points', 'team_a_goal_difference', 'team_a_recent_matches',
    'team_a_recent_win_rate', 'team_a_recent_draw_rate', 'team_a_recent_loss_rate',
    'team_a_recent_points_per_game', 'team_a_overall_matches', 'team_a_overall_win_rate',
    'team_a_overall_draw_rate', 'team_a_overall_loss_rate', 'team_a_overall_points_per_game',
    'team_a_home_venue_matches', 'team_a_home_venue_win_rate', 'team_a_home_venue_draw_rate',
    'team_a_home_venue_loss_rate', 'team_a_home_venue_points_per_game',
    'team_b_away_venue_points_per_game', 'team_b_away_venue_draw_rate', 'team_b_away_venue_matches',
    'team_b_away_venue_loss_rate', 'team_b_away_venue_win_rate',
    'team_a_current_streak', 'team_a_win_streak', 'team_a_unbeaten_streak',
    'team_b_goal_difference_avg', 'team_b_goals_against_avg', 'team_b_goals_against_total',
    'team_b_goals_for_avg', 'team_b_goals_for_total', 'team_b_goal_difference',
    'team_b_overall_draw_rate', 'team_b_overall_loss_rate', 'team_b_overall_matches',
    'team_b_overall_points_per_game', 'team_b_overall_win_rate',
    'team_b_recent_draw_rate', 'team_b_recent_loss_rate', 'team_b_recent_matches',
    'team_b_recent_points_per_game', 'team_b_recent_win_rate',
    'team_b_current_streak', 'team_b_unbeaten_streak', 'team_b_win_streak',
    'team_b_league_position', 'team_b_points',
    'h2h_matches', 'h2h_team_a_wins', 'h2h_draws', 'h2h_team_b_wins',
    'h2h_team_a_win_rate', 'h2h_avg_goals'
]
SELECTED_FEATURE_NAMES = ['team_a_goal_difference', 'team_a_goal_difference_avg', 'team_a_league_position', 'team_a_points', 'team_b_goal_difference', 'team_b_goal_difference_avg', 'team_b_goals_for_avg', 'team_b_league_position', 'team_b_overall_win_rate', 'team_b_points']