"""
Football Data ETL Pipeline with Enhanced Features

This script provides a ETL (Extract, Transform, Load) pipeline for football data
from external APIs. It supports both local and remote storage options, includes retry logic,
and provides extensive data validation.

Key Features:
- Asynchronous data extraction with rate limiting
- Data transformation and feature engineering
- Multiple storage backends (local files, Neon database)
- Retry logic with exponential backoff
- Data quality validation

Usage:
    python football_etl.py

Optimization Notes:
- Uses asyncio for concurrent API requests (recommended: 5-10 concurrent requests max)
- Rate limiting prevents API quota exhaustion (1 second delay between requests)
- File naming convention: {data_type}_{league_id}_{season}.{extension}
- Duplicate request prevention through file existence checks

Storage Rules:
- Raw data: Stored as JSON files with format: {endpoint}_{league_id}_{season}.json
- Processed data: Stored as CSV files with format: {data_type}_cleaned_{league_id}_{season}.csv
- Consolidated data: Includes timestamp to avoid overwrites
"""

import json
import pandas as pd
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import time
from functools import wraps
from sqlalchemy import create_engine, text
from config import (
    API_KEY, BASE_URL, LEAGUE_IDS, SEASONS,
    RAW_DATA_PATH, PROCESSED_DATA_PATH, RATE_LIMIT_DELAY, STORE,
    NEON_DATABASE_URL
)

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('etl_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure directories exist
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Constants for optimization
MAX_CONCURRENT_REQUESTS = 5  # Optimal number for most APIs
RETRY_ATTEMPTS = 3
BACKOFF_FACTOR = 2


def retry_async(max_attempts: int = RETRY_ATTEMPTS, backoff_factor: float = BACKOFF_FACTOR):
    """
    Decorator for async functions to implement retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = backoff_factor ** attempt
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")

            raise last_exception

        return wrapper

    return decorator


def retry_sync(max_attempts: int = RETRY_ATTEMPTS, backoff_factor: float = BACKOFF_FACTOR):
    """
    Decorator for sync functions to implement retry logic with exponential backoff.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = backoff_factor ** attempt
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")

            raise last_exception

        return wrapper

    return decorator


class DatabaseManager:
    """
    Manages database connections and operations for remote storage.
    Supports Neon (PostgreSQL) database for storing football data.
    """

    def __init__(self, database_url: str):
        """
        Initialize database manager with connection URL.

        Args:
            database_url: PostgreSQL connection string for Neon database
        """
        self.database_url = database_url
        self.engine = None

    def connect(self):
        """Establish database connection."""
        try:
            self.engine = create_engine(self.database_url)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Successfully connected to Neon database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        if not self.engine:
            self.connect()

        create_fixtures_table = """
        CREATE TABLE IF NOT EXISTS fixtures (
            fixture_id BIGINT PRIMARY KEY,
            match_date TIMESTAMP,
            season INTEGER,
            league_id INTEGER,
            league_name VARCHAR(100),
            round VARCHAR(50),
            status VARCHAR(10),
            home_team_id INTEGER,
            home_team_name VARCHAR(100),
            home_goals INTEGER,
            home_winner BOOLEAN,
            away_team_id INTEGER,
            away_team_name VARCHAR(100),
            away_goals INTEGER,
            away_winner BOOLEAN,
            match_result CHAR(1),
            venue_name VARCHAR(200),
            referee VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        create_standings_table = """
        CREATE TABLE IF NOT EXISTS standings (
            id SERIAL PRIMARY KEY,
            league_id INTEGER,
            league_name VARCHAR(100),
            season INTEGER,
            team_id INTEGER,
            team_name VARCHAR(100),
            rank INTEGER,
            points INTEGER,
            goal_diff INTEGER,
            form VARCHAR(10),
            played INTEGER,
            wins INTEGER,
            draws INTEGER,
            losses INTEGER,
            goals_for INTEGER,
            goals_against INTEGER,
            home_played INTEGER,
            home_wins INTEGER,
            home_draws INTEGER,
            home_losses INTEGER,
            home_goals_for INTEGER,
            home_goals_against INTEGER,
            away_played INTEGER,
            away_wins INTEGER,
            away_draws INTEGER,
            away_losses INTEGER,
            away_goals_for INTEGER,
            away_goals_against INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(league_id, season, team_id)
        );
        """

        create_team_stats_table = """
        CREATE TABLE IF NOT EXISTS team_statistics (
            id SERIAL PRIMARY KEY,
            team_id INTEGER,
            league_id INTEGER,
            season INTEGER,
            team_name VARCHAR(100),
            matches_played_total INTEGER,
            wins_total INTEGER,
            draws_total INTEGER,
            losses_total INTEGER,
            goals_for_total INTEGER,
            goals_against_total INTEGER,
            goals_for_avg_total DECIMAL(4,2),
            goals_against_avg_total DECIMAL(4,2),
            clean_sheets_total INTEGER,
            failed_to_score_total INTEGER,
            form VARCHAR(20),
            extracted_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(team_id, league_id, season)
        );
        """

        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_fixtures_table))
                conn.execute(text(create_standings_table))
                conn.execute(text(create_team_stats_table))
                conn.commit()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    @retry_sync()
    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        """
        Save DataFrame to database table with retry logic.

        Args:
            df: DataFrame to save
            table_name: Target table name
            if_exists: How to behave if table exists ('append', 'replace', 'fail')
        """
        if not self.engine:
            self.connect()

        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            logger.info(f"Successfully saved {len(df)} rows to {table_name}")
        except Exception as e:
            logger.error(f"Failed to save data to {table_name}: {e}")
            raise


class FootballDataExtractor:
    """
    Handles data extraction from Football API with rate limiting, error handling, and retry logic.

    Key features:
    - Async HTTP requests with session management
    - Rate limiting to prevent API quota exhaustion
    - Retry logic with exponential backoff
    - Request deduplication through file existence checks
    """

    def __init__(self, api_key: str, base_url: str = BASE_URL, max_concurrent: int = MAX_CONCURRENT_REQUESTS):
        """
        Initialize the data extractor.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API
            max_concurrent: Maximum number of concurrent requests (optimization parameter)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"x-apisports-key": api_key}
        self.session = None
        self.semaphore = asyncio.Semaphore(max_concurrent)  # Limit concurrent requests

    async def __aenter__(self):
        """Async context manager entry - creates HTTP session."""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes HTTP session."""
        if self.session:
            await self.session.close()

    @retry_async()
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Make an async API request with error handling and retry logic.

        Uses semaphore to limit concurrent requests for API optimization.
        Implements rate limiting to prevent quota exhaustion.

        Args:
            endpoint: API endpoint
            params: Request parameters
        Returns:
            JSON response data or None if failed after all retries
        """
        url = f"{self.base_url}/{endpoint}"

        async with self.semaphore:  # Limit concurrent requests
            try:
                # Rate limiting - prevent API quota exhaustion
                await asyncio.sleep(RATE_LIMIT_DELAY)

                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Successfully fetched {endpoint} with params {params}")
                        return data
                    elif response.status == 429:  # Rate limit exceeded
                        logger.warning(f"Rate limit exceeded for {endpoint}. Waiting longer...")
                        await asyncio.sleep(RATE_LIMIT_DELAY * 3)  # Wait longer on rate limit
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
                    else:
                        logger.error(f"API request failed: {response.status} - {endpoint}")
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
            except Exception as e:
                logger.error(f"Request exception for {endpoint}: {str(e)}")
                raise

    async def extract_fixtures(self, league_id: int, season: int) -> Optional[Dict]:
        """
        Extract fixtures data for a specific league and season.

        Fixtures contain match information including teams, scores, dates, and results.
        This is typically the largest dataset and forms the base for ML models.

        Args:
            league_id: League identifier (e.g., 39 for Premier League)
            season: Season year (e.g., 2023)
        Returns:
            Fixtures data or None if extraction failed
        """
        params = {"league": league_id, "season": season}
        return await self._make_request("fixtures", params)

    async def extract_team_statistics(self, team_id: int, league_id: int, season: int) -> Optional[Dict]:
        """
        Extract comprehensive team statistics for ML feature engineering.

        Includes goals, wins, losses, home/away performance, etc.
        Critical for creating predictive features.

        Args:
            team_id: Team identifier
            league_id: League identifier
            season: Season year
        Returns:
            Team statistics data or None if extraction failed
        """
        params = {"team": team_id, "league": league_id, "season": season}
        return await self._make_request("teams/statistics", params)

    async def extract_standings(self, league_id: int, season: int) -> Optional[Dict]:
        """
        Extract league standings/table data.

        Provides current position, points, goal difference - useful for team strength indicators.

        Args:
            league_id: League identifier
            season: Season year
        Returns:
            Standings data or None if extraction failed
        """
        params = {"league": league_id, "season": season}
        return await self._make_request("standings", params)

    async def extract_head_to_head(self, team1_id: int, team2_id: int) -> Optional[Dict]:
        """
        Extract historical head-to-head data between two teams.

        Useful for ML models that consider historical matchup performance.

        Args:
            team1_id: First team identifier
            team2_id: Second team identifier
        Returns:
            Head-to-head data or None if extraction failed
        """
        params = {"h2h": f"{team1_id}-{team2_id}"}
        return await self._make_request("fixtures/headtohead", params)


class FootballDataTransformer:
    """
    Handles data transformation and feature engineering for ML pipeline.

    Key responsibilities:
    - Clean raw API data into structured DataFrames
    - Handle missing values and data type conversions
    - Create derived features for ML models
    - Ensure data consistency across different endpoints
    """

    @staticmethod
    def clean_fixtures_data(raw_fixtures: Dict) -> pd.DataFrame:
        """
        Clean and structure fixtures data into a pandas DataFrame.

        This is the core dataset for match prediction models.
        Handles nested JSON structure and missing values gracefully.

        Args:
            raw_fixtures: Raw fixtures data from API
        Returns:
            Cleaned fixtures DataFrame with standardized columns
        """
        if not raw_fixtures or 'response' not in raw_fixtures:
            logger.warning("No fixtures data to clean")
            return pd.DataFrame()

        fixtures_list = []

        for fixture in raw_fixtures['response']:
            try:
                # Extract basic match information
                fixture_clean = {
                    # Primary identifiers
                    'fixture_id': fixture['fixture']['id'],
                    'match_date': pd.to_datetime(fixture['fixture']['date']),
                    'season': fixture['league']['season'],
                    'league_id': fixture['league']['id'],
                    'league_name': fixture['league']['name'],
                    'round': fixture['league']['round'],
                    'status': fixture['fixture']['status']['short'],

                    # Home team information - critical for ML features
                    'home_team_id': fixture['teams']['home']['id'],
                    'home_team_name': fixture['teams']['home']['name'],
                    'home_goals': fixture['goals']['home'],
                    'home_winner': fixture['teams']['home']['winner'],

                    # Away team information - critical for ML features
                    'away_team_id': fixture['teams']['away']['id'],
                    'away_team_name': fixture['teams']['away']['name'],
                    'away_goals': fixture['goals']['away'],
                    'away_winner': fixture['teams']['away']['winner'],

                    # Target variable for ML models
                    'match_result': FootballDataTransformer._determine_match_result(
                        fixture['goals']['home'],
                        fixture['goals']['away']
                    ),

                    # Additional context features
                    'venue_name': fixture['fixture']['venue']['name'] if fixture['fixture']['venue'] else None,
                    'referee': fixture['fixture']['referee']
                }

                fixtures_list.append(fixture_clean)

            except KeyError as e:
                logger.warning(f"Missing key in fixture data: {e} - Skipping fixture")
                continue

        df = pd.DataFrame(fixtures_list)

        # Data quality improvements
        if not df.empty:
            # Convert date column to datetime if not already
            df['match_date'] = pd.to_datetime(df['match_date'])

            # Sort by match date for time series consistency
            df = df.sort_values('match_date').reset_index(drop=True)

        logger.info(f"✅ Cleaned {len(df)} fixtures")
        return df

    @staticmethod
    def clean_team_statistics(raw_stats: Dict, team_id: int, league_id: int, season: int) -> Dict:
        """
        Clean team statistics data for ML feature engineering.

        Extracts comprehensive team performance metrics that serve as
        predictive features for match outcome models.

        Args:
            raw_stats: Raw team statistics from API
            team_id: Team identifier
            league_id: League identifier
            season: Season year
        Returns:
            Cleaned team statistics dictionary
        """
        if not raw_stats or 'response' not in raw_stats:
            logger.warning(f"No team statistics for team {team_id}")
            return {}

        stats = raw_stats['response']

        try:
            clean_stats = {
                # Identifiers
                'team_id': team_id,
                'league_id': league_id,
                'season': season,
                'team_name': stats['team']['name'],

                # Match statistics - base metrics
                'matches_played_total': stats['fixtures']['played']['total'],
                'matches_played_home': stats['fixtures']['played']['home'],
                'matches_played_away': stats['fixtures']['played']['away'],

                # Win statistics - performance indicators
                'wins_total': stats['fixtures']['wins']['total'],
                'wins_home': stats['fixtures']['wins']['home'],
                'wins_away': stats['fixtures']['wins']['away'],

                # Draw statistics
                'draws_total': stats['fixtures']['draws']['total'],
                'draws_home': stats['fixtures']['draws']['home'],
                'draws_away': stats['fixtures']['draws']['away'],

                # Loss statistics
                'losses_total': stats['fixtures']['loses']['total'],
                'losses_home': stats['fixtures']['loses']['home'],
                'losses_away': stats['fixtures']['loses']['away'],

                # Goal statistics - critical ML features
                'goals_for_total': stats['goals']['for']['total']['total'],
                'goals_for_home': stats['goals']['for']['total']['home'],
                'goals_for_away': stats['goals']['for']['total']['away'],
                'goals_for_avg_total': stats['goals']['for']['average']['total'],
                'goals_for_avg_home': stats['goals']['for']['average']['home'],
                'goals_for_avg_away': stats['goals']['for']['average']['away'],

                'goals_against_total': stats['goals']['against']['total']['total'],
                'goals_against_home': stats['goals']['against']['total']['home'],
                'goals_against_away': stats['goals']['against']['total']['away'],
                'goals_against_avg_total': stats['goals']['against']['average']['total'],
                'goals_against_avg_home': stats['goals']['against']['average']['home'],
                'goals_against_avg_away': stats['goals']['against']['average']['away'],

                # Defensive statistics
                'clean_sheets_total': stats['clean_sheet']['total'],
                'clean_sheets_home': stats['clean_sheet']['home'],
                'clean_sheets_away': stats['clean_sheet']['away'],

                'failed_to_score_total': stats['failed_to_score']['total'],
                'failed_to_score_home': stats['failed_to_score']['home'],
                'failed_to_score_away': stats['failed_to_score']['away'],

                # Form indicator - recent performance
                'form': stats.get('form', ''),

                # Metadata
                'extracted_at': datetime.now()
            }

            return clean_stats

        except KeyError as e:
            logger.warning(f"Missing key in team statistics for team {team_id}: {e}")
            return {}

    @staticmethod
    def clean_standings_data(raw_standings: Dict) -> pd.DataFrame:
        """
        Clean standings data into a pandas DataFrame.

        Provides team ranking and performance metrics at a point in time.
        Useful for creating team strength indicators.

        Args:
            raw_standings: Raw standings data from API
        Returns:
            Cleaned standings DataFrame
        """
        if not raw_standings or 'response' not in raw_standings:
            logger.warning("No standings data to clean")
            return pd.DataFrame()

        standings_list = []

        for league_data in raw_standings['response']:
            league_info = league_data['league']

            # Usually first element contains main standings (not group stage)
            for standing in league_info['standings'][0]:
                standing_clean = {
                    # League information
                    'league_id': league_info['id'],
                    'league_name': league_info['name'],
                    'season': league_info['season'],

                    # Team information
                    'team_id': standing['team']['id'],
                    'team_name': standing['team']['name'],

                    # Position metrics - key ML features
                    'rank': standing['rank'],
                    'points': standing['points'],
                    'goal_diff': standing['goalsDiff'],

                    # Additional context
                    'group': standing.get('group', 'Regular'),
                    'form': standing.get('form', ''),
                    'status': standing.get('status', ''),
                    'description': standing.get('description', ''),

                    # Overall match statistics from standings
                    'played': standing['all']['played'],
                    'wins': standing['all']['win'],
                    'draws': standing['all']['draw'],
                    'losses': standing['all']['lose'],
                    'goals_for': standing['all']['goals']['for'],
                    'goals_against': standing['all']['goals']['against'],

                    # Home performance metrics
                    'home_played': standing['home']['played'],
                    'home_wins': standing['home']['win'],
                    'home_draws': standing['home']['draw'],
                    'home_losses': standing['home']['lose'],
                    'home_goals_for': standing['home']['goals']['for'],
                    'home_goals_against': standing['home']['goals']['against'],

                    # Away performance metrics
                    'away_played': standing['away']['played'],
                    'away_wins': standing['away']['win'],
                    'away_draws': standing['away']['draw'],
                    'away_losses': standing['away']['lose'],
                    'away_goals_for': standing['away']['goals']['for'],
                    'away_goals_against': standing['away']['goals']['against'],
                }

                standings_list.append(standing_clean)

        df = pd.DataFrame(standings_list)

        # Sort by rank for consistency
        if not df.empty:
            df = df.sort_values(['league_id', 'season', 'rank']).reset_index(drop=True)

        logger.info(f"✅ Cleaned standings for {len(df)} teams")
        return df

    @staticmethod
    def _determine_match_result(home_goals: Optional[int], away_goals: Optional[int]) -> Optional[str]:
        """
        Determine match result from goals scored.

        This creates the target variable for ML classification models.

        Args:
            home_goals: Goals scored by home team
            away_goals: Goals scored by away team
        Returns:
            Match result ('H' for home win, 'D' for draw, 'A' for away win) or None if incomplete
        """
        if home_goals is None or away_goals is None:
            return None  # Match not completed yet

        if home_goals > away_goals:
            return 'H'  # Home win
        elif home_goals < away_goals:
            return 'A'  # Away win
        else:
            return 'D'  # Draw


class FootballDataLoader:
    """
    Handles data loading to various storage formats and locations.

    Supports multiple storage backends:
    - Local file storage (JSON for raw data, CSV for processed data)
    todo: Remote database storage (Neon PostgreSQL)

    File naming conventions prevent duplicate API requests:
    - Raw: {endpoint}_{league_id}_{season}.json
    - Processed: {data_type}_cleaned_{league_id}_{season}.csv
    - Consolidated: {data_type}_consolidated_{timestamp}.csv
    """

    def __init__(self, storage_types: List[str] = None):
        """
        Initialize data loader with storage configuration.

        Args:
            storage_types: List of storage types ('local', 'remote')
        """
        self.storage_types = storage_types or ['local']
        self.db_manager = None

        # todo: Remote database storage using Neon
        if 'remote' in self.storage_types:
            self.db_manager = DatabaseManager(NEON_DATABASE_URL)
            self.db_manager.connect()
            self.db_manager.create_tables()

    @retry_sync()
    def save_raw_data(self, data: Dict, filename: str, league_id: int, season: int) -> str:
        """
        Save raw API data to JSON file with league and season in filename.

        File naming prevents duplicate API requests by checking existence before extraction.

        Args:
            data: Raw data from API
            filename: Base filename (e.g., 'fixtures', 'standings')
            league_id: League identifier for filename
            season: Season year for filename
        Returns:
            Full path of saved file
        """
        if 'local' not in self.storage_types:
            return ""

        filepath = RAW_DATA_PATH / f"{filename}_{league_id}_{season}.json"

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Use default=str to handle datetime objects
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Saved raw data to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"❌ Failed to save raw data to {filepath}: {e}")
            raise

    @retry_sync()
    def save_processed_data(self, df: pd.DataFrame, filename: str, league_id: int, season: int) -> str:
        """
        Save processed DataFrame to CSV and/or database.

        Args:
            df: Processed DataFrame
            filename: Base filename
            league_id: League identifier for filename
            season: Season year for filename
        Returns:
            Full path of saved file (for local storage)
        """
        saved_path = ""

        # Local storage
        if 'local' in self.storage_types:
            filepath = PROCESSED_DATA_PATH / f"{filename}_{league_id}_{season}.csv"
            try:
                df.to_csv(filepath, index=False, encoding='utf-8')
                logger.info(f"✅ Saved processed data to {filepath} ({len(df)} rows)")
                saved_path = str(filepath)
            except Exception as e:
                logger.error(f"❌ Failed to save processed data to {filepath}: {e}")
                raise

        # todo: Remote database storage using Neon
        if 'remote' in self.storage_types and self.db_manager:
            try:
                # Map filename to table name
                table_mapping = {
                    'fixtures_cleaned': 'fixtures',
                    'standings_cleaned': 'standings',
                    'team_stats_cleaned': 'team_statistics'
                }

                table_name = table_mapping.get(filename, filename.replace('_cleaned', ''))

                # Prepare DataFrame for database insertion
                df_copy = df.copy()

                # Handle specific data type conversions for database
                if 'match_date' in df_copy.columns:
                    df_copy['match_date'] = pd.to_datetime(df_copy['match_date'])

                # Save to database with upsert logic
                self.db_manager.save_dataframe(df_copy, table_name, if_exists='append')
                logger.info(f"✅ Saved {len(df)} rows to database table {table_name}")

            except Exception as e:
                logger.error(f"❌ Failed to save to database: {e}")
                # Don't raise here to allow local storage to succeed

        return saved_path

    @retry_sync()
    def save_consolidated_data(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save consolidated DataFrame to CSV with timestamp to avoid overwrites.

        Args:
            df: DataFrame to save
            filename: Base filename
        Returns:
            Full path of saved file
        """
        if 'local' not in self.storage_types:
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = PROCESSED_DATA_PATH / f"{filename}_consolidated_{timestamp}.csv"

        try:
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"✅ Saved consolidated data: {filepath} ({len(df)} rows)")
            return str(filepath)

        except Exception as e:
            logger.error(f"❌ Failed to save consolidated data: {e}")
            raise

    @staticmethod
    def check_file_exists(filename: str, league_id: int, season: int, data_type: str = 'raw') -> bool:
        """
        Check if data file already exists to avoid duplicate API requests.

        This is critical for optimization - prevents unnecessary API calls
        and helps manage API quota limits.

        Args:
            filename: Base filename
            league_id: League identifier
            season: Season year
            data_type: 'raw' or 'processed'
        Returns:
            True if file exists, False otherwise
        """
        if data_type == 'raw':
            filepath = RAW_DATA_PATH / f"{filename}_{league_id}_{season}.json"
        else:
            filepath = PROCESSED_DATA_PATH / f"{filename}_{league_id}_{season}.csv"

        exists = filepath.exists()
        if exists:
            logger.info(f"File already exists: {filepath}")

        return exists


# ETL Functions for Airflow Integration
async def extract_football_data(
        league_ids: List[int] = None,
        seasons: List[int] = None,
        storage_types: List[str] = None
) -> Dict[str, Any]:
    """
    Extract all required data for multiple leagues and seasons.

    This is the main extraction function designed for Airflow tasks.
    Uses semaphore-based concurrency control for optimal API usage.

    Optimization strategy:
    - Concurrent requests limited to MAX_CONCURRENT_REQUESTS (5-10 recommended)
    - Rate limiting with RATE_LIMIT_DELAY between requests
    - File existence checks prevent duplicate API calls
    - Retry logic handles temporary failures

    Args:
        league_ids: List of league identifiers (defaults to config)
        seasons: List of season years (defaults to config)
        storage_types: List of storage types ('local', 'remote')
    Returns:
        Dictionary containing all extracted data and summary statistics
    """
    if league_ids is None:
        league_ids = LEAGUE_IDS
    if seasons is None:
        seasons = SEASONS
    if storage_types is None:
        storage_types = ['local']

    # Initialize data loader
    loader = FootballDataLoader(storage_types)

    all_data = {
        'fixtures': {},
        'team_statistics': {},
        'standings': {},
        'extraction_summary': []
    }

    # Track extraction statistics for monitoring
    total_requests = 0
    successful_requests = 0
    skipped_requests = 0

    async with FootballDataExtractor(API_KEY) as extractor:

        # Extract fixtures and standings for each league/season combination
        for league_id in league_ids:
            for season in seasons:
                logger.info(f"Processing league {league_id}, season {season}")

                # Extract fixtures - core dataset for ML models
                if loader.check_file_exists('fixtures', league_id, season, 'raw'):
                    logger.info(
                        f"Skipping fixtures extraction for league {league_id}, season {season} - already exists")
                    skipped_requests += 1
                else:
                    logger.info(f"Extracting fixtures for league {league_id}, season {season}")
                    total_requests += 1

                    fixtures_data = await extractor.extract_fixtures(league_id, season)

                    if fixtures_data:
                        all_data['fixtures'][f"{league_id}_{season}"] = fixtures_data
                        loader.save_raw_data(fixtures_data, 'fixtures', league_id, season)
                        successful_requests += 1

                        logger.info(f"Extracted {len(fixtures_data.get('response', []))} fixtures")

                # Extract standings - team strength indicators
                if not loader.check_file_exists('standings', league_id, season, 'raw'):
                    logger.info(f"Extracting standings for league {league_id}, season {season}")
                    total_requests += 1

                    standings_data = await extractor.extract_standings(league_id, season)

                    if standings_data:
                        all_data['standings'][f"{league_id}_{season}"] = standings_data
                        loader.save_raw_data(standings_data, 'standings', league_id, season)
                        successful_requests += 1
                else:
                    skipped_requests += 1

                # Extract team statistics for ML feature engineering
                # Only extract if we have fixtures data (to get team IDs)
                fixtures_key = f"{league_id}_{season}"
                if fixtures_key in all_data['fixtures'] or loader.check_file_exists('fixtures', league_id, season,
                                                                                    'raw'):

                    # Load fixtures data if not in memory
                    if fixtures_key not in all_data['fixtures']:
                        fixtures_file = RAW_DATA_PATH / f"fixtures_{league_id}_{season}.json"
                        if fixtures_file.exists():
                            with open(fixtures_file, 'r') as f:
                                all_data['fixtures'][fixtures_key] = json.load(f)

                    fixtures_data = all_data['fixtures'].get(fixtures_key)
                    if fixtures_data and 'response' in fixtures_data:
                        # Extract unique team IDs from fixtures
                        team_ids = set()
                        for fixture in fixtures_data['response']:
                            team_ids.add(fixture['teams']['home']['id'])
                            team_ids.add(fixture['teams']['away']['id'])

                        logger.info(f"Extracting team statistics for {len(team_ids)} teams")

                        # Extract team statistics with concurrency control
                        for team_id in team_ids:
                            if not loader.check_file_exists(f'team_stats_{team_id}', league_id, season, 'raw'):
                                total_requests += 1

                                team_stats = await extractor.extract_team_statistics(team_id, league_id, season)

                                if team_stats:
                                    key = f"{team_id}_{league_id}_{season}"
                                    all_data['team_statistics'][key] = team_stats
                                    loader.save_raw_data(team_stats, f'team_stats_{team_id}', league_id, season)
                                    successful_requests += 1
                            else:
                                skipped_requests += 1

                # Add extraction summary for monitoring
                all_data['extraction_summary'].append({
                    'league_id': league_id,
                    'season': season,
                    'fixtures_count': len(
                        fixtures_data.get('response', [])) if 'fixtures_data' in locals() and fixtures_data else 0,
                    'extraction_time': datetime.now()
                })

    # Log extraction statistics
    logger.info(f"Extraction completed:")
    logger.info(f"  Total API requests: {total_requests}")
    logger.info(f"  Successful requests: {successful_requests}")
    logger.info(f"  Skipped requests (already existed): {skipped_requests}")
    logger.info(
        f"  Success rate: {(successful_requests / total_requests * 100):.1f}%" if total_requests > 0 else "  No new requests needed")

    return all_data


def transform_football_data(
        raw_data: Dict[str, Any] = None,
        storage_types: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Transform all raw data into cleaned DataFrames suitable for ML models.

    This function handles the T (Transform) phase of ETL.
    Designed to work with data in memory or load from files.

    Args:
        raw_data: Dictionary containing all raw data (if None, loads from files)
        storage_types: List of storage types for saving transformed data
    Returns:
        Dictionary of cleaned DataFrames ready for ML pipeline
    """
    transformed_data = {}
    transformer = FootballDataTransformer()

    if storage_types is None:
        storage_types = ['local']

    loader = FootballDataLoader(storage_types)

    # If no raw_data provided, load from existing files
    # This allows the transform step to run independently
    if raw_data is None:
        logger.info("No raw data provided, loading from existing files")
        raw_data = {'fixtures': {}, 'standings': {}, 'team_statistics': {}}

        # Load raw data from files for each league/season combination
        for league_id in LEAGUE_IDS:
            for season in SEASONS:
                # Load fixtures
                fixtures_file = RAW_DATA_PATH / f"fixtures_{league_id}_{season}.json"
                if fixtures_file.exists():
                    with open(fixtures_file, 'r') as f:
                        raw_data['fixtures'][f"{league_id}_{season}"] = json.load(f)

                # Load standings
                standings_file = RAW_DATA_PATH / f"standings_{league_id}_{season}.json"
                if standings_file.exists():
                    with open(standings_file, 'r') as f:
                        raw_data['standings'][f"{league_id}_{season}"] = json.load(f)

                # Load team statistics
                for team_stats_file in RAW_DATA_PATH.glob(f"team_stats_*_{league_id}_{season}.json"):
                    team_id = team_stats_file.stem.split('_')[2]  # Extract team_id from filename
                    key = f"{team_id}_{league_id}_{season}"
                    with open(team_stats_file, 'r') as f:
                        raw_data['team_statistics'][key] = json.load(f)

    # Transform fixtures - core dataset for ML models
    for key, fixtures_data in raw_data['fixtures'].items():
        league_id, season = key.split('_')

        if loader.check_file_exists('fixtures_cleaned', int(league_id), int(season), 'processed'):
            logger.info(f"Skipping fixtures transformation for {key} - already exists")
            continue

        logger.info(f"Transforming fixtures for {key}")
        fixtures_df = transformer.clean_fixtures_data(fixtures_data)

        if not fixtures_df.empty:
            transformed_data[f'fixtures_{key}'] = fixtures_df
            loader.save_processed_data(fixtures_df, 'fixtures_cleaned', int(league_id), int(season))

    # Transform standings - team strength indicators
    for key, standings_data in raw_data['standings'].items():
        league_id, season = key.split('_')

        if loader.check_file_exists('standings_cleaned', int(league_id), int(season), 'processed'):
            logger.info(f"Skipping standings transformation for {key} - already exists")
            continue

        logger.info(f"Transforming standings for {key}")
        standings_df = transformer.clean_standings_data(standings_data)

        if not standings_df.empty:
            transformed_data[f'standings_{key}'] = standings_df
            loader.save_processed_data(standings_df, 'standings_cleaned', int(league_id), int(season))

    # Transform team statistics - ML features
    team_stats_list = []
    for key, team_stats_data in raw_data['team_statistics'].items():
        team_id, league_id, season = key.split('_')

        logger.info(f"Transforming team statistics for {key}")
        clean_stats = transformer.clean_team_statistics(
            team_stats_data,
            int(team_id),
            int(league_id),
            int(season)
        )

        if clean_stats:
            team_stats_list.append(clean_stats)

    # Consolidate team statistics into single DataFrame
    if team_stats_list:
        team_stats_df = pd.DataFrame(team_stats_list)
        transformed_data['team_statistics_all'] = team_stats_df

        # Save team statistics grouped by league and season for easier access
        for (league_id, season), group in team_stats_df.groupby(['league_id', 'season']):
            loader.save_processed_data(group, 'team_stats_cleaned', league_id, season)

    logger.info(f"Transformation completed for {len(transformed_data)} datasets")
    return transformed_data


def load_football_data(
        transformed_data: Dict[str, pd.DataFrame],
        storage_types: List[str] = None
) -> Dict[str, str]:
    """
    Load all transformed data to final storage locations.

    This handles the L (Load) phase of ETL.
    Creates consolidated datasets for ML model training.

    Args:
        transformed_data: Dictionary of transformed DataFrames
        storage_types: List of storage types ('local', 'remote')
    Returns:
        Dictionary of file paths where data was saved
    """
    if storage_types is None:
        storage_types = ['local']

    loader = FootballDataLoader(storage_types)
    saved_files = {}

    for data_name, df in transformed_data.items():
        if df.empty:
            logger.warning(f"Skipping empty dataset: {data_name}")
            continue

        # Save consolidated data with timestamp to avoid overwrites
        logger.info(f"Loading {data_name} ({len(df)} rows)")
        filepath = loader.save_consolidated_data(df, data_name)
        if filepath:
            saved_files[data_name] = filepath

    logger.info(f"Load phase completed - {len(saved_files)} files saved")
    return saved_files


# Utility functions for data loading and validation
def load_processed_data(data_type: str, league_id: int = None, season: int = None) -> pd.DataFrame:
    """
    Load processed data from CSV files for analysis or ML model training.

    Supports loading specific league/season or all available data.

    Args:
        data_type: Type of data ('fixtures', 'standings', 'team_stats')
        league_id: League identifier (if None, loads all)
        season: Season year (if None, loads all)
    Returns:
        Loaded DataFrame with requested data
    """
    dfs = []

    if league_id is not None and season is not None:
        # Load specific league and season
        filepath = PROCESSED_DATA_PATH / f"{data_type}_cleaned_{league_id}_{season}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            dfs.append(df)
        else:
            logger.warning(f"File not found: {filepath}")
    else:
        # Load all files matching pattern
        pattern = f"{data_type}_cleaned_*.csv"
        for filepath in PROCESSED_DATA_PATH.glob(pattern):
            df = pd.read_csv(filepath)
            dfs.append(df)

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"✅ Loaded {len(combined_df)} rows of {data_type} data")
        return combined_df
    else:
        logger.warning(f"❌ No {data_type} data found")
        return pd.DataFrame()


def validate_data_quality(df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
    """
    Validate data quality and return comprehensive summary statistics.

    Critical for ensuring ML model input quality.
    Checks for missing values, duplicates, and data-specific validations.

    Args:
        df: DataFrame to validate
        data_type: Type of data being validated
    Returns:
        Dictionary with validation results and quality metrics
    """
    validation_results = {
        'data_type': data_type,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'validation_timestamp': datetime.now()
    }

    # Data-specific validations for ML model readiness
    if data_type == 'fixtures':
        validation_results.update({
            'completed_matches': len(df[df['match_result'].notna()]),
            'pending_matches': len(df[df['match_result'].isna()]),
            'unique_teams': len(set(df['home_team_id'].tolist() + df['away_team_id'].tolist())),
            'leagues_covered': df['league_id'].nunique(),
            'seasons_covered': df['season'].nunique(),
            'match_results_distribution': df['match_result'].value_counts().to_dict(),
            'date_range': {
                'earliest': df['match_date'].min() if 'match_date' in df.columns else None,
                'latest': df['match_date'].max() if 'match_date' in df.columns else None
            }
        })

    elif data_type == 'standings':
        validation_results.update({
            'unique_teams': df['team_id'].nunique(),
            'leagues_covered': df['league_id'].nunique(),
            'seasons_covered': df['season'].nunique(),
            'avg_points': df['points'].mean(),
            'avg_goal_diff': df['goal_diff'].mean(),
            'position_range': {
                'min_rank': df['rank'].min(),
                'max_rank': df['rank'].max()
            }
        })

    elif data_type == 'team_statistics':
        validation_results.update({
            'unique_teams': df['team_id'].nunique(),
            'leagues_covered': df['league_id'].nunique(),
            'seasons_covered': df['season'].nunique(),
            'avg_goals_for': df['goals_for_avg_total'].mean() if 'goals_for_avg_total' in df.columns else None,
            'avg_goals_against': df[
                'goals_against_avg_total'].mean() if 'goals_against_avg_total' in df.columns else None,
            'avg_matches_played': df['matches_played_total'].mean() if 'matches_played_total' in df.columns else None
        })

    # Check for critical data quality issues
    critical_issues = []
    if validation_results['duplicate_rows'] > 0:
        critical_issues.append(f"Found {validation_results['duplicate_rows']} duplicate rows")

    missing_threshold = 0.1  # 10% missing data threshold
    for col, missing_count in validation_results['missing_values'].items():
        if missing_count > len(df) * missing_threshold:
            critical_issues.append(
                f"Column '{col}' has {missing_count}/{len(df)} missing values ({missing_count / len(df) * 100:.1f}%)")

    validation_results['critical_issues'] = critical_issues
    validation_results['data_quality_score'] = 100 - len(critical_issues) * 10  # Simple scoring system

    logger.info(f"✅ Data validation completed for {data_type}")
    if critical_issues:
        logger.warning(f"Critical data quality issues found: {critical_issues}")

    return validation_results


async def run_etl_pipeline(
        league_ids: List[int] = None,
        seasons: List[int] = None,
        storage_types: List[str] = None
):
    """
    Main ETL pipeline function that orchestrates the entire process.

    This is the main entry point for running the complete ETL pipeline.
    Includes comprehensive logging, error handling, and performance monitoring.

    Pipeline phases:
    1. Extract - Fetch data from Football API with retry logic
    2. Transform - Clean and structure data for ML models
    3. Load - Save to configured storage backends
    4. Validate - Check data quality and generate reports

    Args:
        league_ids: List of league identifiers to process
        seasons: List of seasons to process
        storage_types: List of storage types ('local', 'remote')
    """
    start_time = datetime.now()
    logger.info("Starting Football Data ETL Pipeline")
    logger.info("Configuration:")
    logger.info(f"  Leagues: {league_ids or LEAGUE_IDS}")
    logger.info(f"  Seasons: {seasons or SEASONS}")
    logger.info(f"  Storage: {storage_types or ['local']}")

    pipeline_stats = {
        'start_time': start_time,
        'phases_completed': [],
        'errors': [],
        'data_summary': {}
    }

    try:
        # Phase 1: Extract data from API
        logger.info("Phase 1: Extracting data from API")
        phase_start = datetime.now()

        raw_data = await extract_football_data(league_ids, seasons, storage_types)

        phase_duration = datetime.now() - phase_start
        pipeline_stats['phases_completed'].append({
            'phase': 'extract',
            'duration': phase_duration,
            'data_points': sum(len(data.get('response', [])) for data in raw_data.get('fixtures', {}).values())
        })
        logger.info(f"Extraction completed in {phase_duration}")

        # Phase 2: Transform data
        logger.info("Phase 2: Transforming data")
        phase_start = datetime.now()

        transformed_data = transform_football_data(raw_data, storage_types)

        phase_duration = datetime.now() - phase_start
        pipeline_stats['phases_completed'].append({
            'phase': 'transform',
            'duration': phase_duration,
            'datasets_created': len(transformed_data)
        })
        logger.info(f"Transformation completed in {phase_duration}")

        # Phase 3: Load data to storage
        logger.info("Phase 3: Loading data to storage")
        phase_start = datetime.now()

        saved_files = load_football_data(transformed_data, storage_types)

        phase_duration = datetime.now() - phase_start
        pipeline_stats['phases_completed'].append({
            'phase': 'load',
            'duration': phase_duration,
            'files_saved': len(saved_files)
        })
        logger.info(f"Loading completed in {phase_duration}")

        # Phase 4: Validate data quality
        logger.info("Phase 4: Validating data quality")
        validation_results = {}

        for data_name, df in transformed_data.items():
            if not df.empty:
                data_type = data_name.split('_')[0]  # Extract type from name
                validation_results[data_name] = validate_data_quality(df, data_type)

        pipeline_stats['data_summary'] = validation_results

        # Final summary
        end_time = datetime.now()
        total_duration = end_time - start_time

        logger.info("ETL Pipeline completed successfully!")
        logger.info("Pipeline Summary:")
        logger.info(f"  Total execution time: {total_duration}")
        logger.info(f"  Phases completed: {len(pipeline_stats['phases_completed'])}")
        logger.info(f"  Files saved: {len(saved_files)}")
        logger.info(f"  Storage types: {storage_types or ['local']}")

        # Log saved files
        if saved_files:
            logger.info("Files saved:")
            for data_type, filepath in saved_files.items():
                logger.info(f"   - {data_type}: {filepath}")

        # Log validation summary
        logger.info("Data Quality Summary:")
        for data_name, validation in validation_results.items():
            quality_score = validation.get('data_quality_score', 0)
            logger.info(f"   - {data_name}: {validation['total_rows']} rows, Quality Score: {quality_score}/100")

        return {
            'success': True,
            'pipeline_stats': pipeline_stats,
            'saved_files': saved_files,
            'validation_results': validation_results
        }

    except Exception as e:
        end_time = datetime.now()
        total_duration = end_time - start_time

        logger.error(f"ETL Pipeline failed after {total_duration}")
        logger.error(f"Error: {str(e)}")

        pipeline_stats['errors'].append({
            'error': str(e),
            'timestamp': datetime.now()
        })

        return {
            'success': False,
            'pipeline_stats': pipeline_stats,
            'error': str(e)
        }


def main():
    """
    Main function with command-line argument parsing.

    Supports command-line execution with configurable parameters:
    - Storage types (local, remote)
    - League IDs
    - Seasons

    Example usage:
        python football_etl.py
    """
    # Run the async ETL pipeline with parsed arguments

    result = asyncio.run(run_etl_pipeline(
        league_ids=LEAGUE_IDS,
        seasons=SEASONS,
        storage_types=STORE
    ))

    if result['success']:
        logger.info("Pipeline completed successfully")
        exit(0)
    else:
        logger.error("Pipeline failed")
        exit(1)


if __name__ == "__main__":
    main()
