"""
Feature Calculator for Football Match Prediction

This module computes ML features for match prediction using processed ETL data.
Features include team form, head-to-head records, goals statistics, and venue performance.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import pandas as pd
from config import FEATURE_NAMES, PROCESSED_DATA_PATH, SELECTED_FEATURE_NAMES

logger = logging.getLogger(__name__)


class FeatureCalculator:
    """
    Computes ML features for match prediction using processed ETL data.

    For production predictions with inputs: league_id, team_a_name, team_b_name.
    Loads processed CSV files from ETL pipeline and computes real-time features.
    """

    def __init__(self, lookback_matches: int = 10):
        """
        Initialize feature calculator with processed data.

        :param lookback_matches: Number of recent matches to consider for form calculation.
        """
        self.lookback_matches = lookback_matches  # Number of recent matches for form
        # #[?] take only recent matches during same season ?
        self.fixtures_data: Dict[Tuple[int, int], pd.DataFrame] = {}  # (league_id, season) -> fixtures_df
        self.standings_data: Dict[Tuple[int, int], pd.DataFrame] = {}  # (league_id, season) -> standings_df
        self.team_stats_data: Dict[Tuple[int, int], pd.DataFrame] = {}  # (league_id, season) -> team_stats_df
        # #[?] not used
        self._load_processed_data()

    def _load_processed_data(self) -> None:
        """
        Load all processed CSV files into memory for fast feature computation.

        Loads fixtures, standings, and team statistics from ETL processed data.
        """
        processed_path = Path(PROCESSED_DATA_PATH)

        # Load fixtures data (core dataset)
        for fixtures_file in processed_path.glob("fixtures_cleaned_*.csv"):
            try:
                # Extract league_id and season from filename: fixtures_cleaned_39_2023.csv
                parts = fixtures_file.stem.split('_')
                league_id, season = int(parts[2]), int(parts[3])

                df = pd.read_csv(fixtures_file)
                df['match_date'] = pd.to_datetime(df['match_date'])  # Ensure datetime format
                # Filter only completed matches with results
                df = df[df['match_result'].notna()].copy()
                df = df.sort_values('match_date').reset_index(drop=True)  # Sort by date

                self.fixtures_data[(league_id, season)] = df
                logger.info(f"Loaded {len(df)} fixtures for league {league_id}, season {season}")

            except (IndexError, ValueError) as e:
                logger.warning(f"Could not parse filename {fixtures_file}: {e}")
                continue

        # Load standings data (team strength indicators)
        for standings_file in processed_path.glob("standings_cleaned_*.csv"):
            try:
                parts = standings_file.stem.split('_')
                league_id, season = int(parts[2]), int(parts[3])

                df = pd.read_csv(standings_file)
                self.standings_data[(league_id, season)] = df
                logger.info(f"Loaded standings for league {league_id}, season {season}")

            except (IndexError, ValueError) as e:
                logger.warning(f"Could not parse standings filename {standings_file}: {e}")
                continue

        # Load team statistics data (detailed team metrics)
        for team_stats_file in processed_path.glob("team_stats_cleaned_*.csv"):
            try:
                parts = team_stats_file.stem.split('_')
                league_id, season = int(parts[3]), int(parts[4])

                df = pd.read_csv(team_stats_file)
                self.team_stats_data[(league_id, season)] = df
                logger.info(f"Loaded team stats for league {league_id}, season {season}")

            except (IndexError, ValueError) as e:
                logger.warning(f"Could not parse team stats filename {team_stats_file}: {e}")
                continue

        logger.info(f"Loaded data for {len(self.fixtures_data)} league-season combinations")

    @staticmethod
    def get_processed_features(data_df: pd.DataFrame, tasks=['scaling', 'imputation']):
        pass
        #todo

    def compute_training_features(self, fixtures_df: pd.DataFrame, league_id: int, season: int) -> pd.DataFrame:
        """
        Compute features for all matches in training dataset.

        :param fixtures_df: DataFrame with fixture data.
        :param league_id: League identifier.
        :param season: Season year.
        :return: features_df with feature names and target (match_result) as columns
        """
        features_list = []

        # Process each completed match
        completed_matches = fixtures_df[fixtures_df['match_result'].notna()].copy()

        for idx, match in completed_matches.iterrows():
            try:
                # Get historical data up to this match date
                historical_data = fixtures_df[fixtures_df['match_date'] < match['match_date']].copy()

                if len(historical_data) < 10:  # Need minimum historical data
                    continue

                # Compute features using historical data only
                match_features_dict = self._compute_match_features_historical(
                    historical_data,
                    match['home_team_id'],
                    match['away_team_id'],
                    league_id,
                    season,
                    return_dict=True,
                    feature_names=FEATURE_NAMES,
                )

                if match_features_dict is not None:
                    # add targets
                    # Target encoding: H=0, D=1, A=2
                    target_map = {'H': 0, 'D': 1, 'A': 2}
                    match_features_dict['match_result'] = target_map[match['match_result']]
                    features_list.append(match_features_dict)


            except Exception as e:
                logger.warning(f"Error processing match {match['fixture_id']}: {e}")
                continue

        if not features_list:
            logger.warning("No valid training features computed")
            return pd.DataFrame(), np.empty(0)

        # Create DataFrame with feature names as columns
        features_df = pd.DataFrame(features_list, columns=FEATURE_NAMES+['match_result'])

        logger.info(f"Computed training features: {features_df.shape[0]} samples, {features_df.shape[1]} features")
        return features_df

    def _get_latest_season_data(self, league_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the most recent season data for a league.

        :param league_id: League identifier.
        :return: Dictionary with fixtures data and season info, or None if not found.
        """
        available_seasons = [season for lid, season in self.fixtures_data.keys() if lid == league_id]
        if not available_seasons:
            return None

        latest_season = max(available_seasons)
        return {
            'fixtures': self.fixtures_data[(league_id, latest_season)],
            'season': latest_season
        }

    def _get_team_id(self, fixtures_df: pd.DataFrame, team_name: str) -> Optional[int]:
        """
        Get team ID from team name using fuzzy matching.

        :param fixtures_df: DataFrame with fixture data.
        :param team_name: Team name to search for.
        :return: Team ID or None if not found.
        """
        # Try exact match first
        home_exact = fixtures_df[fixtures_df['home_team_name'] == team_name]
        if not home_exact.empty:
            return home_exact.iloc[0]['home_team_id']

        away_exact = fixtures_df[fixtures_df['away_team_name'] == team_name]
        if not away_exact.empty:
            return away_exact.iloc[0]['away_team_id']

        # Try partial match (case insensitive)
        home_partial = fixtures_df[fixtures_df['home_team_name'].str.contains(team_name, case=False, na=False)]
        if not home_partial.empty:
            return home_partial.iloc[0]['home_team_id']

        away_partial = fixtures_df[fixtures_df['away_team_name'].str.contains(team_name, case=False, na=False)]
        if not away_partial.empty:
            return away_partial.iloc[0]['away_team_id']

        return None

    def _compute_team_features(self, fixtures_df: pd.DataFrame, team_id: int, venue: str,
                               season: int, league_id: int) -> Dict[str, float]:
        """
        Compute comprehensive team features for ML model.

        :param fixtures_df: DataFrame with fixture data. #[?] one line is one match information
        :param team_id: Team identifier.
        :param venue: 'home' or 'away' for venue-specific features.
        :param season: Season year.
        :param league_id: League identifier.
        :return: Dictionary of team features.
        """
        features = {}

        # Get team matches (both home and away)
        team_matches = fixtures_df[
            ((fixtures_df['home_team_id'] == team_id) |
             (fixtures_df['away_team_id'] == team_id)) &
            (fixtures_df['match_result'].notna())
            ].copy() #[?] why do we have fixtures_df['match_result'] missing values ? #[?] why copy() ?

        if team_matches.empty:
            return self._get_default_team_features()

        # Recent form (last N matches)
        recent_matches = team_matches.tail(self.lookback_matches)

        # Overall performance features
        features.update(self._compute_performance_features(team_matches, team_id, 'overall'))

        # Recent form features
        features.update(self._compute_performance_features(recent_matches, team_id, 'recent'))

        # Venue-specific features #[?] performance features when team is home/away
        venue_features = self._compute_venue_features(team_matches, team_id, venue)
        features.update(venue_features)

        # Goal statistics
        goal_features = self._compute_goal_features(team_matches, team_id)
        features.update(goal_features)

        # Form streak features
        streak_features = self._compute_streak_features(recent_matches, team_id)
        features.update(streak_features)

        # League position features (if standings available)
        if (league_id, season) in self.standings_data:
            position_features = self._compute_league_position_features(league_id, season, team_id)
            features.update(position_features)

        return features

    def _compute_performance_features(self, matches_df: pd.DataFrame, team_id: int, prefix: str) -> Dict[str, float]:
        """
        Compute team performance features (wins, draws, losses).

        :param matches_df: DataFrame with team matches.
        :param team_id: Team identifier.
        :param prefix: Feature name prefix ('overall' or 'recent').
        :return: Dictionary of performance features.
        """
        if matches_df.empty:
            return {
                f'{prefix}_matches': 0,
                f'{prefix}_win_rate': 0,
                f'{prefix}_draw_rate': 0,
                f'{prefix}_loss_rate': 0,
                f'{prefix}_points_per_game': 0
            }

        total_matches = len(matches_df)
        wins = 0
        draws = 0
        losses = 0

        # H → Home win (the home team won the match), D → Draw (the match ended in a tie),
        # A → Away win (the away team won the match)
        #[?] Use groupby and size to compute them => matches_df[matches_df['home_team_id'] == team_id].groupby('match_result').size().transform(lambda x: x/sum(x))
        for _, match in matches_df.iterrows():
            if match['home_team_id'] == team_id:
                # Team was playing at home
                if match['match_result'] == 'H':
                    wins += 1
                elif match['match_result'] == 'D':
                    draws += 1
                else:
                    losses += 1
            else:
                # Team was playing away
                if match['match_result'] == 'A':
                    wins += 1
                elif match['match_result'] == 'D':
                    draws += 1
                else:
                    losses += 1

        result = {
            f'{prefix}_matches': total_matches,
            f'{prefix}_win_rate': wins / total_matches if total_matches > 0 else 0,
            f'{prefix}_draw_rate': draws / total_matches if total_matches > 0 else 0,
            f'{prefix}_loss_rate': losses / total_matches if total_matches > 0 else 0,
            f'{prefix}_points_per_game': (wins * 3 + draws) / total_matches if total_matches > 0 else 0
        }

        return result ##[?] why don't we have team_a_home_venue_points_per_game, team_b_away_venue_points_per_game

    def _compute_venue_features(self, matches_df: pd.DataFrame, team_id: int, venue: str) -> Dict[str, float]:
        """
        Compute venue-specific performance features.

        :param matches_df: DataFrame with team matches.
        :param team_id: Team identifier.
        :param venue: 'home' or 'away'.
        :return: Dictionary of venue-specific features.
        """
        if venue == 'home':
            venue_matches = matches_df[matches_df['home_team_id'] == team_id]
        else:
            venue_matches = matches_df[matches_df['away_team_id'] == team_id]

        return self._compute_performance_features(venue_matches, team_id, f'{venue}_venue')

    def _compute_goal_features(self, matches_df: pd.DataFrame, team_id: int) -> Dict[str, float]:
        """
        Compute goal-related features (goals for/against, averages).

        :param matches_df: DataFrame with team matches.
        :param team_id: Team identifier.
        :return: Dictionary of goal features.
        """
        #[?] goals_for = matches_df[(matches_df['home_goals'].notna()) & (matches_df['away_goals'].notna())]['home_goals']
        # goals_for_avg = goals_for.sum() / goals_for.shape[0]
        # goals_total = goals_for.sum()
        # goals_away = matches_df[(matches_df['home_goals'].notna()) & (matches_df['away_goals'].notna())]['away_goals']
        # goal_difference_avg = abs((goals_for.sum() - goals_away.sum())) / len(matches_df[(matches_df['home_goals'].notna()) & (matches_df['away_goals'].notna())])
        if matches_df.empty:
            return {
                'goals_for_avg': 0,
                'goals_against_avg': 0,
                'goal_difference_avg': 0,
                'goals_for_total': 0,
                'goals_against_total': 0
            }

        goals_for = []
        goals_against = []

        for _, match in matches_df.iterrows():
            if pd.isna(match['home_goals']) or pd.isna(match['away_goals']):
                continue

            if match['home_team_id'] == team_id:
                # Team was home
                goals_for.append(match['home_goals'])
                goals_against.append(match['away_goals'])
            else:
                # Team was away
                goals_for.append(match['away_goals'])
                goals_against.append(match['home_goals'])

        if not goals_for:
            return {
                'goals_for_avg': 0,
                'goals_against_avg': 0,
                'goal_difference_avg': 0,
                'goals_for_total': 0,
                'goals_against_total': 0
            }

        avg_goals_for = np.mean(goals_for)
        avg_goals_against = np.mean(goals_against)

        return {
            'goals_for_avg': avg_goals_for,
            'goals_against_avg': avg_goals_against,
            'goal_difference_avg': avg_goals_for - avg_goals_against,
            'goals_for_total': sum(goals_for),
            'goals_against_total': sum(goals_against)
        }

    def _compute_streak_features(self, recent_matches: pd.DataFrame, team_id: int) -> Dict[str, float]:
        """
        Compute form streak features (current winning/losing streak).

        :param recent_matches: DataFrame with recent matches.
        :param team_id: Team identifier.
        :return: Dictionary of streak features.
        """
        if recent_matches.empty:
            return {'current_streak': 0, 'win_streak': 0, 'unbeaten_streak': 0}

        # Sort by date to get chronological order
        recent_matches = recent_matches.sort_values('match_date')

        # Calculate results from team perspective
        results = []
        for _, match in recent_matches.iterrows():
            if match['home_team_id'] == team_id:
                result = match['match_result']  # H, D, A
                if result == 'H':
                    results.append('W')
                elif result == 'D':
                    results.append('D')
                else:
                    results.append('L')
            else:
                result = match['match_result']
                if result == 'A':
                    results.append('W')
                elif result == 'D':
                    results.append('D')
                else:
                    results.append('L')

        # Calculate streaks
        current_streak = self._calculate_current_streak(results)
        win_streak = self._calculate_specific_streak(results, 'W')
        unbeaten_streak = self._calculate_unbeaten_streak(results)

        return {
            'current_streak': current_streak,
            'win_streak': win_streak,
            'unbeaten_streak': unbeaten_streak
        }

    def _compute_h2h_features(self, fixtures_df: pd.DataFrame, team_a_id: int, team_b_id: int) -> Dict[str, float]:
        """
        Compute head-to-head features between two teams.

        :param fixtures_df: DataFrame with fixture data.
        :param team_a_id: First team identifier.
        :param team_b_id: Second team identifier.
        :return: Dictionary of head-to-head features.
        """
        """
        [?] 'h2h_matches' is total head-to-head matches number,
            'h2h_team_a_wins', team_a win number
            'h2h_draws': draws,
            'h2h_team_b_wins', team_b_wins number
            'h2h_team_a_win_rate', head-to-head team_a win rate
            'h2h_avg_goals': head-to-head average goals
        """
        # Find matches between these two teams
        h2h_matches = fixtures_df[
            ((fixtures_df['home_team_id'] == team_a_id) & (fixtures_df['away_team_id'] == team_b_id)) |
            ((fixtures_df['home_team_id'] == team_b_id) & (fixtures_df['away_team_id'] == team_a_id))
            ].copy()

        if h2h_matches.empty:
            return {
                'h2h_matches': 0, 'h2h_team_a_wins': 0, 'h2h_draws': 0, 'h2h_team_b_wins': 0,
                'h2h_team_a_win_rate': 0, 'h2h_avg_goals': 0
            }
        # [?] Use groupby:
        # h2h_team_a_wins_df = h2h_matches[(h2h_matches['match_result'] == 'H') & (h2h_matches['home_team_id'] == team_a_id)]
        # h2h_team_a_wins = h2h_team_a_wins_df['home_goals'].sum()
        total_matches = len(h2h_matches)
        team_a_wins = 0
        draws = 0
        team_b_wins = 0
        total_goals = 0

        for _, match in h2h_matches.iterrows():
            # Count goals
            if pd.notna(match['home_goals']) and pd.notna(match['away_goals']):
                total_goals += match['home_goals'] + match['away_goals']

            # Count results from team_a perspective
            if match['home_team_id'] == team_a_id:
                # Team A was home
                if match['match_result'] == 'H':
                    team_a_wins += 1
                elif match['match_result'] == 'D':
                    draws += 1
                else:
                    team_b_wins += 1
            else:
                # Team A was away
                if match['match_result'] == 'A':
                    team_a_wins += 1
                elif match['match_result'] == 'D':
                    draws += 1
                else:
                    team_b_wins += 1

        return {
            'h2h_matches': total_matches,
            'h2h_team_a_wins': team_a_wins,
            'h2h_draws': draws,
            'h2h_team_b_wins': team_b_wins,
            'h2h_team_a_win_rate': team_a_wins / total_matches if total_matches > 0 else 0,
            'h2h_avg_goals': total_goals / total_matches if total_matches > 0 else 0
        }

    def _compute_league_position_features(self, league_id: int, season: int, team_id: int) -> Dict[str, float]:
        """
        Compute league position-based features from standings.

        :param league_id: League identifier.
        :param season: Season year.
        :param team_id: Team identifier.
        :return: Dictionary of position features.
        """
        #[?] should be computed one time, no repeated search for same team
        standings_df = self.standings_data.get((league_id, season))
        if standings_df is None:
            return {'league_position': 0, 'points': 0, 'goal_difference': 0}

        team_standing = standings_df[standings_df['team_id'] == team_id]
        if team_standing.empty:
            return {'league_position': 0, 'points': 0, 'goal_difference': 0}

        team_row = team_standing.iloc[0]
        return {
            'league_position': team_row.get('rank', 0),
            'points': team_row.get('points', 0),
            'goal_difference': team_row.get('goal_diff', 0)
        }

    def _compute_league_context_features(self, league_id: int, season: int) -> Dict[str, float]:
        """
        Compute league-level context features.

        :param league_id: League identifier.
        :param season: Season year.
        :return: Dictionary of league context features.
        """
        return {
            'league_id': float(league_id),
            'season': float(season),
            'is_top_league': float(league_id in [39, 140, 135, 78, 61])  # Top 5 European leagues
        }

    def _get_default_team_features(self) -> Dict[str, float]:
        """
        Get default team features when no data is available.
        Returns all possible team features to match FEATURE_NAMES structure.

        :return: Dictionary of default features.
        """
        return {
            # Overall performance (5 features)
            'overall_matches': 0, 'overall_win_rate': 0, 'overall_draw_rate': 0, 'overall_loss_rate': 0,
            'overall_points_per_game': 0,
            # Recent form (5 features)
            'recent_matches': 0, 'recent_win_rate': 0, 'recent_draw_rate': 0, 'recent_loss_rate': 0,
            'recent_points_per_game': 0,
            # Home venue (5 features)
            'home_venue_matches': 0, 'home_venue_win_rate': 0, 'home_venue_draw_rate': 0,
            'home_venue_loss_rate': 0, 'home_venue_points_per_game': 0,
            # Away venue (5 features)
            'away_venue_matches': 0, 'away_venue_win_rate': 0, 'away_venue_draw_rate': 0,
            'away_venue_loss_rate': 0, 'away_venue_points_per_game': 0,
            # Goal statistics (5 features)
            'goals_for_avg': 0, 'goals_against_avg': 0, 'goal_difference_avg': 0,
            'goals_for_total': 0, 'goals_against_total': 0,
            # Streak features (3 features)
            'current_streak': 0, 'win_streak': 0, 'unbeaten_streak': 0,
            # League position (3 features)
            'league_position': 0, 'points': 0, 'goal_difference': 0
        }

    def _calculate_current_streak(self, results: List[str]) -> int:
        """
        Calculate current form streak (positive for wins, negative for losses).

        :param results: List of recent results ('W', 'D', 'L').
        :return: Current streak value.
        """
        if not results:
            return 0

        streak = 0
        last_result = results[-1]

        # Count backwards while same result
        for result in reversed(results):
            if result == last_result:
                if result == 'W':
                    streak += 1
                elif result == 'L':
                    streak -= 1
                # Draws don't change streak value significantly
            else:
                break

        return streak

    def _calculate_specific_streak(self, results: List[str], target_result: str) -> int:
        """
        Calculate streak of specific result type.

        :param results: List of recent results.
        :param target_result: Target result type ('W', 'D', 'L').
        :return: Streak count.
        """
        streak = 0
        for result in reversed(results):
            if result == target_result:
                streak += 1
            else:
                break
        return streak

    def _calculate_unbeaten_streak(self, results: List[str]) -> int:
        """
        Calculate unbeaten streak (wins + draws).

        :param results: List of recent results.
        :return: Unbeaten streak count.
        """
        streak = 0
        #[?] why reversed
        for result in reversed(results):
            if result in ['W', 'D']:
                streak += 1
            else:
                break
        return streak

    def get_available_teams(self, league_id: int) -> List[str]:
        """
        Get list of available team names for a specific league.

        :param league_id: League identifier.
        :return: List of team names.
        """
        teams = set()

        for (lid, season), fixtures_df in self.fixtures_data.items():
            if lid == league_id:
                teams.update(fixtures_df['home_team_name'].unique())
                teams.update(fixtures_df['away_team_name'].unique())

        return sorted(list(teams))

    def compute_match_features(self, league_id: int, team_a_name: str, team_b_name: str,
                               match_date: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Compute match features using only historical data before the match date.
        #[?] for predict
        """
        current_season_data = self._get_latest_season_data(league_id)
        if current_season_data is None:
            return None

        fixtures_df = current_season_data['fixtures']
        season = current_season_data['season']
        logger.info(f"Got {len(fixtures_df)} fixtures for league {league_id}, season {season}")
        # Determine cutoff date with proper timezone handling
        if match_date is None:
            if fixtures_df['match_date'].dt.tz is not None:
                cutoff_date = pd.Timestamp.now(tz='UTC')
            else:
                cutoff_date = pd.Timestamp.now()
        else:
            cutoff_date = pd.to_datetime(match_date)
            if fixtures_df['match_date'].dt.tz is not None and cutoff_date.tz is None:
                cutoff_date = cutoff_date.tz_localize('UTC')

        # Use only historical data before the match
        historical_data = fixtures_df[fixtures_df['match_date'] < cutoff_date].copy()

        if len(historical_data) < 10:
            logger.warning(f"Insufficient historical data: {len(historical_data)} matches")
            return None

        # Get team IDs
        team_a_id = self._get_team_id(historical_data, team_a_name)
        team_b_id = self._get_team_id(historical_data, team_b_name)

        if team_a_id is None or team_b_id is None:
            logger.warning(f"Could not find team IDs for {team_a_name} or {team_b_name}")
            return None

        # Use unified feature computation logic (same as training)
        result = self._compute_match_features_historical(
            historical_df=historical_data, home_team_id=team_a_id, away_team_id=team_b_id,
            league_id=league_id, season=season, feature_names=SELECTED_FEATURE_NAMES,
        )

        return result

    def _compute_match_features_historical(self, historical_df: pd.DataFrame, home_team_id: int,
                                           away_team_id: int, league_id: int, season: int, feature_names: list,
                                           return_dict: bool = False, ) -> Optional[np.ndarray]:
        """
        Compute features using only historical data (for training and prediction).
        Returns features aligned with feature names ordering.

        :param return_dict: If True, returns dictionary instead of numpy array for DataFrame creation
        """
        try:
            # Initialize feature dictionary with all expected features
            all_features = {}

            # Home team (team_a) features
            home_features = self._compute_team_features(historical_df, home_team_id, 'home', season, league_id)
            for key, value in home_features.items():
                all_features[f'team_a_{key}'] = value

            # Away team (team_b) features
            away_features = self._compute_team_features(historical_df, away_team_id, 'away', season, league_id)
            for key, value in away_features.items():
                all_features[f'team_b_{key}'] = value

            # Head-to-head features
            h2h_features = self._compute_h2h_features(historical_df, home_team_id, away_team_id)
            all_features.update(h2h_features)

            # League context features
            league_features = self._compute_league_context_features(league_id, season)
            all_features.update(league_features)
            # feature_names before or after feature selection
            #[?] todo: each feature computed separately, we can compute them based on feature_names before or
            # after feature selection


            # Return dictionary for DataFrame creation if requested
            if return_dict:
                # Ensure all 61 feature names are present
                feature_dict = {}
                for feature_name in feature_names:
                    feature_dict[feature_name] = all_features.get(feature_name, 0.0) #[?] 0.0 or nan ?
                return feature_dict

            # Create feature vector aligned with feature_names
            feature_vector = []
            missing_features = []

            for feature_name in feature_names:
                if feature_name in all_features:
                    feature_vector.append(all_features[feature_name])
                else:
                    feature_vector.append(0.0)  # Default value for missing features
                    missing_features.append(feature_name)

            if missing_features:
                logger.info(f"Missing features (filled with 0.0): {missing_features}")

            feature_vector = np.array(feature_vector)
            if len(feature_vector) != len(feature_names):
                logger.error(f"Final feature vector length: {len(feature_vector)} (expected: {len(feature_names)})")

            # Verify feature vector length matches expected 61 features
            if len(feature_vector) != len(feature_names):
                logger.error(
                    f"Feature vector length mismatch: got {len(feature_vector)}, expected {len(feature_names)}")
                return None

            return feature_vector

        except Exception as e:
            logger.error(f"Error in historical feature computation: {e}")
            return None
