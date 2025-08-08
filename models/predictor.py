"""
Football Match Predictor

This module provides the main prediction interface for football match outcomes.
Handles model loading, feature computation, and prediction generation for production use.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import joblib
from config import LEAGUE_IDS, SELECTED_FEATURE_NAMES
from models.feature_calculator import FeatureCalculator

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('models_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FootballPredictor:
    """
    Main predictor class for football match outcome prediction.

    Handles model loading, feature computation, and prediction generation.
    Supports multiple models and provides confidence scores and explanations.
    """

    def __init__(self, model_version: str = "latest"):
        """
        Initialize the football predictor with trained models.

        :param model_version: Version of the model to load ('latest' or specific version).
        """
        self.model_version = model_version
        self.model = None  # Main ML model
        self.scaler = None  # Feature scaler
        self.feature_calculator = FeatureCalculator()  # Feature computation engine
        self.model_metadata: Dict[str, Any] = {}  # Model information and performance
        self.class_names = ['Home Win', 'Draw', 'Away Win']  # Prediction classes
        self.model_path = Path("models/saved_models")  # Path to saved models

        # Load trained model and preprocessing components
        self._load_model()

        logger.info(f"FootballPredictor initialized with model version: {model_version}")

    def predict(self, league_id: int, team_a_name: str, team_b_name: str) -> Dict[str, Any]:
        """
        Predict match outcome for given teams (main production method).

        :param league_id: League identifier (e.g., 39 for Premier League).
        :param team_a_name: Home team name.
        :param team_b_name: Away team name.
        :return: Dictionary containing prediction results, probabilities, and metadata.
        """
        try:
            # Validate inputs
            if not self._validate_inputs(league_id, team_a_name, team_b_name):
                return self._get_error_response("Invalid input parameters")

            # Compute features using feature calculator
            features = self.feature_calculator.compute_match_features(league_id, team_a_name, team_b_name)

            if features is None:
                return self._get_error_response("Could not compute features - insufficient data")

            # Scale features if scaler is available
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)

            # Generate predictions
            prediction_proba = self.model.predict_proba(features_scaled)[0]  # Get probabilities
            predicted_class = np.argmax(prediction_proba)  # Get most likely outcome
            confidence = float(prediction_proba[predicted_class])  # Confidence score

            # Create comprehensive response
            response = {
                'success': True,
                'prediction': {
                    'outcome': self.class_names[predicted_class],
                    'outcome_code': self._get_outcome_code(predicted_class),
                    'confidence': confidence,
                    'probabilities': {
                        'home_win': float(prediction_proba[0]),
                        'draw': float(prediction_proba[1]),
                        'away_win': float(prediction_proba[2])
                    }
                },
                'match_info': {
                    'home_team': team_a_name,
                    'away_team': team_b_name,
                    'league_id': league_id,
                    'league_name': self._get_league_name(league_id)
                },
                'model_info': {
                    'version': self.model_version,
                    'features_used': len(features),
                    'model_accuracy': self.model_metadata.get('accuracy', 'Unknown')
                },
                'explanation': self._generate_explanation(features, predicted_class, confidence),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(
                f"Prediction generated: {team_a_name} vs {team_b_name} -> {self.class_names[predicted_class]} ({confidence:.2f})")
            return response

        except Exception as e:
            logger.error(f"Error during prediction for {team_a_name} vs {team_b_name}: {e}")
            return self._get_error_response(f"Prediction failed: {str(e)}")

    def predict_batch(self, matches: List[Tuple[int, str, str]]) -> List[Dict[str, Any]]:
        """
        Predict outcomes for multiple matches efficiently.

        :param matches: List of tuples (league_id, team_a_name, team_b_name).
        :return: List of prediction results for each match.
        """
        results = []

        logger.info(f"Processing batch prediction for {len(matches)} matches")

        for i, (league_id, team_a_name, team_b_name) in enumerate(matches):
            try:
                result = self.predict(league_id, team_a_name, team_b_name)
                result['batch_index'] = i
                results.append(result)

            except Exception as e:
                logger.error(f"Error in batch prediction for match {i}: {e}")
                error_result = self._get_error_response(f"Batch prediction failed: {str(e)}")
                error_result['batch_index'] = i
                results.append(error_result)

        logger.info(f"Batch prediction completed: {len(results)} results generated")
        return results

    def get_prediction_confidence_threshold(self, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Filter predictions based on confidence threshold.

        :param min_confidence: Minimum confidence score to accept predictions.
        :return: List of high-confidence predictions.
        """
        # This would be used with batch predictions
        # Implementation depends on specific use case
        pass

    def get_model_performance(self) -> Dict[str, Any]:
        """
        Get model performance metrics and information.

        :return: Dictionary with model performance data.
        """
        return {
            'model_version': self.model_version,
            'metadata': self.model_metadata,
            'class_names': self.class_names,
            'model_type': str(type(self.model).__name__) if self.model else 'Unknown'
        }

    def get_available_teams(self, league_id: int) -> List[str]:
        """
        Get available teams for prediction in a specific league.

        :param league_id: League identifier.
        :return: List of team names available for prediction.
        """
        return self.feature_calculator.get_available_teams(league_id)

    def validate_team_names(self, league_id: int, team_a_name: str, team_b_name: str) -> Dict[str, bool]:
        """
        Validate if team names exist in the specified league.

        :param league_id: League identifier.
        :param team_a_name: Home team name.
        :param team_b_name: Away team name.
        :return: Dictionary indicating validity of each team name.
        """
        available_teams = self.get_available_teams(league_id)

        return {
            'team_a_valid': any(team_a_name.lower() in team.lower() for team in available_teams),
            'team_b_valid': any(team_b_name.lower() in team.lower() for team in available_teams),
            'available_teams': available_teams
        }

    def _load_model(self) -> None:
        """
        Load trained model and preprocessing components from disk.

        Loads the ML model, feature scaler, and model metadata.
        """
        try:
            # Create model directory if it doesn't exist
            self.model_path.mkdir(parents=True, exist_ok=True)

            if self.model_version == "latest":
                # Find the most recent model file
                model_files = list(self.model_path.glob("*_model_*.pkl"))
                if not model_files:
                    raise FileNotFoundError("No model files found")

                # Sort by modification time and get the latest
                latest_model_file = max(model_files, key=lambda p: p.stat().st_mtime)
                model_file = latest_model_file
            else:
                # Load specific version
                model_file = self.model_path / f"model_{self.model_version}.pkl"
                if not model_file.exists():
                    raise FileNotFoundError(f"Model version {self.model_version} not found")

            # Load main model
            self.model = joblib.load(model_file)
            logger.info(f"Loaded model from {model_file}")

            # Load scaler if available
            scaler_file = self.model_path / f"scaler_{self.model_version}.pkl"
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
                logger.info(f"Loaded scaler from {scaler_file}")

            # Load metadata if available
            metadata_file = self.model_path / f"metadata_{self.model_version}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_file}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Initialize dummy model for development
            self._initialize_dummy_model()

    def _initialize_dummy_model(self) -> None:
        """
        Initialize a dummy model for development/testing when no trained model exists.

        Creates a simple random model that can be used for testing the pipeline.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        logger.warning("No trained model found, initializing dummy model for development")

        # Create dummy model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.scaler = StandardScaler()

        # Fit dummy model with random data matching expected feature count
        feature_count = len(SELECTED_FEATURE_NAMES) or 50  # Default 50 features
        dummy_X = np.random.randn(100, feature_count)
        dummy_y = np.random.randint(0, 3, 100)  # Random classes 0, 1, 2

        self.scaler.fit(dummy_X)
        scaled_X = self.scaler.transform(dummy_X)
        self.model.fit(scaled_X, dummy_y)

        self.model_metadata = {
            'version': 'dummy',
            'accuracy': 0.33,  # Random accuracy
            'created_at': datetime.now().isoformat(),
            'is_dummy': True
        }

        logger.info("Dummy model initialized successfully")

    def _validate_inputs(self, league_id: int, team_a_name: str, team_b_name: str) -> bool:
        """
        Validate input parameters for prediction.

        :param league_id: League identifier.
        :param team_a_name: Home team name.
        :param team_b_name: Away team name.
        :return: True if inputs are valid, False otherwise.
        """
        # Check league ID
        if league_id not in LEAGUE_IDS:
            logger.warning(f"League ID {league_id} not in supported leagues")
            return False

        # Check team names
        if not team_a_name or not team_b_name:
            logger.warning("Team names cannot be empty")
            return False

        if team_a_name.lower() == team_b_name.lower():
            logger.warning("Team names cannot be the same")
            return False

        return True

    def _get_outcome_code(self, predicted_class: int) -> str:
        """
        Convert predicted class index to outcome code.

        :param predicted_class: Predicted class index (0, 1, 2).
        :return: Outcome code ('H', 'D', 'A').
        """
        code_map = {0: 'H', 1: 'D', 2: 'A'}  # Home, Draw, Away
        return code_map.get(predicted_class, 'Unknown')

    def _get_league_name(self, league_id: int) -> str:
        """
        Get league name from league ID.

        :param league_id: League identifier.
        :return: League name or 'Unknown' if not found.
        """
        league_names = {
            39: 'Premier League',
            140: 'La Liga',
            135: 'Serie A',
            78: 'Bundesliga',
            61: 'Ligue 1',
            88: 'Eredivisie',
            94: 'Liga Portugal',
            253: 'MLS',
            71: 'Brasileirão',
            2: 'UEFA Champions League',
            3: 'UEFA Europa League'
        }
        return league_names.get(league_id, f'League {league_id}')

    def _generate_explanation(self, features: np.ndarray, predicted_class: int, confidence: float) -> Dict[str, Any]:
        """
        Generate human-readable explanation for the prediction.

        :param features: Feature vector used for prediction.
        :param predicted_class: Predicted outcome class.
        :param confidence: Prediction confidence score.
        :return: Dictionary with prediction explanation.
        """
        explanation = {
            'outcome': self.class_names[predicted_class],
            'confidence_level': self._get_confidence_level(confidence),
            'key_factors': [],
            'recommendation': self._get_recommendation(confidence)
        }

        # Add key factors based on feature importance (simplified)
        feature_names = SELECTED_FEATURE_NAMES
        if len(feature_names) == len(features):
            # Find most influential features (basic implementation)
            important_indices = np.argsort(np.abs(features))[-5:]  # Top 5 features by magnitude

            for idx in important_indices:
                if idx < len(feature_names):
                    explanation['key_factors'].append({
                        'factor': feature_names[idx],
                        'value': float(features[idx]),
                        'impact': 'positive' if features[idx] > 0 else 'negative'
                    })

        return explanation

    def _get_confidence_level(self, confidence: float) -> str:
        """
        Convert confidence score to human-readable level.

        :param confidence: Confidence score (0-1).
        :return: Confidence level description.
        """
        if confidence >= 0.8:
            return 'Very High'
        elif confidence >= 0.6:
            return 'High'
        elif confidence >= 0.4:
            return 'Medium'
        else:
            return 'Low'

    def _get_recommendation(self, confidence: float) -> str:
        """
        Generate betting/decision recommendation based on confidence.

        :param confidence: Prediction confidence score.
        :return: Recommendation string.
        """
        if confidence >= 0.7:
            return 'Strong prediction - consider for betting'
        elif confidence >= 0.5:
            return 'Moderate confidence - proceed with caution'
        else:
            return 'Low confidence - avoid betting on this match'

    def _get_feature_importance(self, feature_name: str, feature_index: int) -> float:
        """
        Get importance score for a specific feature.

        :param feature_name: Name of the feature.
        :param feature_index: Index of the feature in the vector.
        :return: Importance score (0-1).
        """
        # If model has feature_importances_ attribute (e.g., RandomForest, XGBoost)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if feature_index < len(importances):
                return float(importances[feature_index])

        # Default importance for unknown models
        return 0.5

    def _get_feature_description(self, feature_name: str) -> str:
        """
        Get human-readable description for a feature.

        :param feature_name: Name of the feature.
        :return: Feature description.
        """
        descriptions = {
            'recent_win_rate': 'Recent winning percentage',
            'goals_for_avg': 'Average goals scored per match',
            'goals_against_avg': 'Average goals conceded per match',
            'home_venue_win_rate': 'Home venue winning percentage',
            'away_venue_win_rate': 'Away venue winning percentage',
            'h2h_team_a_win_rate': 'Head-to-head winning percentage',
            'current_streak': 'Current form streak',
            'league_position': 'Current league position'
        }

        return descriptions.get(feature_name, feature_name.replace('_', ' ').title())

    def _get_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Generate standardized error response.

        :param error_message: Error description.
        :return: Error response dictionary.
        """
        return {
            'success': False,
            'error': error_message,
            'prediction': None,
            'timestamp': datetime.now().isoformat()
        }


def predict():
    before = datetime.now()
    predictor = FootballPredictor()
    #  [?] todo, store scaler during training to be reused during prediction
    result = predictor.predict(
        league_id=88,
        team_a_name="Fortuna Sittard",
        team_b_name="GO Ahead Eagles"  # [?]todo: try 'Chelsea' couldn't be found in 2024 league, to fix
    )
    logger.info(result['prediction']['outcome'])
    logger.info(result['prediction']['confidence'])
    logger.info(f'Execution time =  {datetime.now() - before}')


if __name__ == '__main__':
    predict()
