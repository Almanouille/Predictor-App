"""
Football Match Prediction Model Trainer

This module handles the training of ML models for football match prediction.
Supports multiple algorithms, hyperparameter tuning, and model evaluation.
"""
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from lightgbm import LGBMClassifier

import joblib
import pandas as pd
import xgboost as xgb
from config import LEAGUE_IDS, PROCESSED_DATA_PATH, SEASONS
from lazypredict.Supervised import LazyClassifier
from models.feature_calculator import FeatureCalculator
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import (LogisticRegression, RidgeClassifier,
                                  RidgeClassifierCV, SGDClassifier)
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.utils.class_weight import compute_class_weight
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

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


class FootballModelTrainer:
    """
    Handles training of ML models for football match prediction.

    Supports multiple algorithms, cross-validation, hyperparameter tuning,
    and comprehensive model evaluation with performance metrics.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.

        :param random_state: Random seed for reproducible results.
        """
        self.random_state = random_state
        self.feature_calculator = FeatureCalculator()  # Feature computation engine
        self.models: Dict[str, Any] = {}  # Trained models storage
        self.scalers: Dict[str, StandardScaler] = {}  # Feature scalers
        self.training_history: List[Dict[str, Any]] = []  # Training history log
        self.best_model_name: Optional[str] = None  # Best performing model
        self.class_names = ['Home Win', 'Draw', 'Away Win']  # Target classes
        self.save_path = Path("models/saved_models")  # Model save directory

        # Create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Define available model configurations
        self.model_configs = self._get_model_configurations()

        logger.info("FootballModelTrainer initialized")

    def prepare_training_data(self, leagues: List[int] = None, seasons: List[int] = None,
                              test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and testing datasets from processed ETL data.

        :param leagues: List of league IDs to include (defaults to all available).
        :param seasons: List of seasons to include (defaults to all available).
        :param test_size: Proportion of data for testing (0.0-1.0).
        :return: Tuple of (X_train, X_test, y_train, y_test).
        """
        if leagues is None:
            leagues = LEAGUE_IDS
        if seasons is None:
            seasons = SEASONS
        print("prepare training data for ", leagues, seasons)

        logger.info(f"Preparing training data for leagues: {leagues}, seasons: {seasons}")

        all_features = []

        # Load and process data for each league-season combination
        for league_id in leagues:
            for season in seasons:
                try:
                    # Load fixtures data
                    fixtures_file = Path(PROCESSED_DATA_PATH) / f"fixtures_cleaned_{league_id}_{season}.csv"
                    if not fixtures_file.exists():
                        logger.warning(f"Fixtures file not found: {fixtures_file}")
                        continue

                    fixtures_df = pd.read_csv(fixtures_file)
                    fixtures_df['match_date'] = pd.to_datetime(fixtures_df['match_date'])

                    logger.info(f"Processing {len(fixtures_df)} matches for league {league_id}, season {season}")

                    # Compute features for all matches
                    features_df  = self.feature_calculator.compute_training_features(
                        fixtures_df, league_id, season
                    )

                    if len(features_df) > 0:
                        all_features.append(features_df)
                        #all_targets.append(targets)
                        logger.info(f"Added {len(features_df)} training samples from league {league_id}, season {season}")


                except Exception as e:
                    logger.error(f"Error processing league {league_id}, season {season}: {e}")
                    continue

        if not all_features:
            raise ValueError("No training data could be prepared")

        # Combine all data
        data_df = pd.concat(all_features, ignore_index=True)
        logger.info(f'Total matches nb={len(data_df)}')
        breakpoint()
        #todo: check missing value columns in data_df, which strategy to use for each column
        #todo: add scaling and missing value handling, get_processed_data, should be called during train and prediction


        # Split based on season
        train_mask = data_df['season'].isin([2020, 2021, 2022, 2023])
        test_mask = data_df['season'] == 2024


        X_train = data_df[train_mask].drop('match_result', axis=1)
        y_train = data_df[train_mask]['match_result']
        X_test = data_df[test_mask].drop('match_result', axis=1)
        y_test = data_df[test_mask]['match_result']
        logger.info("training data prepared:::::")

        logger.info("Training data prepared:")
        logger.info(f"  Total samples: {len(data_df)}")
        logger.info(f"  Features number: {X_train.shape[1]}")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Test samples: {len(X_test)}")
        logger.info(f"  Class distribution: {np.bincount(y_train)}")

        # Feature selection
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        best_k, best_features, best_score = self.get_best_k(X=X_scaled, y=y_train, k_values=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        #[?]todo: final logging to tell user to update SELECTED_FEATURE_NAMES in config.py
        logger.info(f"Features selection: features number after features selection = {len(best_features)}, best_features = :\n{sorted(best_features)}")
        return X_train[best_features], X_test[best_features], y_train, y_test



    @staticmethod
    def select_k_best_features(X: pd.DataFrame, y: np.ndarray, k: int) -> list:
        """
        Select the top K features using mutual information.

        :param X: Feature DataFrame.
        :param y: Target array.
        :param k: Number of features to select.
        :return: List of selected feature names.
        """
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X, y)
        mask = selector.get_support()
        selected_features = X.columns[mask].tolist()

        for name, score in zip(X.columns[mask], selector.scores_[mask]):
            logger.info(f"Selected feature: {name}, Score: {score:.4f}")

        return selected_features

    def get_best_k(self, X: pd.DataFrame, y: np.ndarray, k_values: list) -> tuple:
        """
        Find the best K value by cross-validating a RandomForest on selected features.
        Reason for RandomForest: Chosen for speed, stability, and ease of use in feature selection.

        :param X: Feature DataFrame.
        :param y: Target array.
        :param k_values: List of K values to try.
        :return: Tuple of (best_k, selected_features, best_f_score).
        """
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        best_score = 0
        best_k = None
        best_features = None

        for k in k_values:
            selected = self.select_k_best_features(X, y, k)
            clf = RandomForestClassifier(random_state=self.random_state)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            score = cross_val_score(clf, X[selected], y, cv=cv, scoring="f1_macro").mean()
            logger.info(f"Tested k={k}, F1 Score={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
                best_features = selected

        return best_k, best_features, best_score


    def run_all_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray) -> list:

        # Run LazyPredict, with scaling and missing value imputation
        # "imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler()),
        # https://github.com/shankarpandala/lazypredict/blob/dev/lazypredict/Supervised.py

        clf = LazyClassifier(verbose=0, ignore_warnings=True, random_state=self.random_state)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        # Sort by F1 Score
        sorted_models = models.sort_values(by="F1 Score", ascending=False)
        logger.info(sorted_models)

        # Extract top 2 model names that exist in your model config
        available_models = list(self.model_configs.keys())
        top3_model_names = sorted_models.index[:3]

        logger.info(f"Selected top 3 models for fine-tuning: {top3_model_names}")
        if not all(item in available_models for item in top3_model_names):
            logger.error(f"Not all top model are in available_models")
        top_available_models = [model for model in top3_model_names if model in available_models]
        logger.info(f"Models to fine tune: {top_available_models}")

        return top_available_models

    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train a single model with specified configuration.

        :param model_name: Name of the model to train (e.g., 'random_forest', 'xgboost').
        :param X_train: Training features.
        :param y_train: Training targets.
        :param X_test: Test features.
        :param y_test: Test targets.
        :param hyperparameter_tuning: Whether to perform hyperparameter tuning.
        :return: Dictionary with training results and metrics.
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.model_configs.keys())}")

        logger.info(f"Training {model_name} model...")
        start_time = datetime.now()

        # Get model configuration
        config = self.model_configs[model_name]

        # Initialize model and scaler
        model = config['model']

        scaler = StandardScaler()

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Handle class imbalance
        if config.get('handle_imbalance', False):
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(zip(np.unique(y_train), class_weights))

            if hasattr(model, 'class_weight'):
                model.set_params(class_weight=class_weight_dict)
            elif hasattr(model, 'scale_pos_weight') and model_name == 'xgboost':
                # XGBoost specific handling
                scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
                model.set_params(scale_pos_weight=scale_pos_weight)

        # Hyperparameter tuning
        if hyperparameter_tuning and 'param_grid' in config:
            logger.info(f"Performing Bayesian hyperparameter tuning for {model_name}")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            search_space = self.convert_param_grid_to_skopt_space(config['param_grid'])
            bayes_search = BayesSearchCV(
                model,
                search_space,
                cv=cv,
                scoring='f1_macro',
                n_jobs=-1,
                n_iter=30,
                verbose=1,
                random_state=self.random_state
            )
            bayes_search.fit(X_train_scaled, y_train)

            best_model = bayes_search.best_estimator_

            logger.info(f"Best parameters for {model_name}: {bayes_search.best_params_}")

        else:
            best_model = model
            best_model.fit(X_train_scaled, y_train)

        # Generate predictions
        y_train_pred = best_model.predict(X_train_scaled)
        y_test_pred = best_model.predict(X_test_scaled)

        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)

        # Cross-validation score, use macro F1 for multi-class classification
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='f1_macro')

        # Training results
        training_time = datetime.now() - start_time
        results = {
            'model_name': model_name,
            'model': best_model,
            'scaler': scaler,
            'metrics': metrics,
            'cv_scores': {
                'mean': float(np.mean(cv_scores)),
                'std': float(np.std(cv_scores)),
                'scores': cv_scores.tolist()
            },
            'training_time': training_time.total_seconds(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat()
        }

        # Store model and scaler
        self.models[model_name] = best_model
        self.scalers[model_name] = scaler

        # Add to training history
        self.training_history.append(results)

        logger.info(f"Model {model_name} training completed:")
        logger.info(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"  Test F1: {metrics['test_f1']:.4f}")
        logger.info(f"  CV Accuracy: {results['cv_scores']['mean']:.4f} (+/- {results['cv_scores']['std'] * 2:.4f})")
        logger.info(f"  Training Time: {training_time.total_seconds():.2f} seconds")

        return results

    @staticmethod
    def convert_param_grid_to_skopt_space(param_grid: Dict[str, list]) -> Dict[str, Any]:
        """ Convert a parameter grid dictionary (list of values) to skopt search space format.

        :param param_grid: Dictionary with parameter names and list of possible values.
        :return: Dictionary with parameter names and skopt space objects (Integer, Real, Categorical).
        """
        skopt_space = {}
        for k, v in param_grid.items():
            if all(isinstance(x, int) for x in v) and len(v) > 1:
                skopt_space[k] = Integer(min(v), max(v))
            elif all(isinstance(x, float) for x in v) and len(v) > 1:
                skopt_space[k] = Real(min(v), max(v))
            else:

                skopt_space[k] = Categorical(v)
        return skopt_space

    def train_top_models(self, X_train: pd.DataFrame, y_train: np.ndarray,
                         X_test: pd.DataFrame, y_test: np.ndarray,
                         hyperparameter_tuning: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Train all available models and compare their performance.

        :param X_train: Training features.
        :param y_train: Training targets.
        :param X_test: Test features.
        :param y_test: Test targets.
        :param hyperparameter_tuning: Whether to perform hyperparameter tuning.
        :return: Dictionary with results for all trained models.
        """
        # Before fine tuning
        top_models = self.run_all_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        all_results = {}

        # Fine tune on top models
        for model_name in top_models:
            try:
                results = self.train_single_model(
                    model_name, X_train, y_train, X_test, y_test, hyperparameter_tuning
                )
                all_results[model_name] = results

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue

        # Find best model based on test f1
        if all_results:
            best_model_name = max(all_results.keys(),
                                  key=lambda k: all_results[k]['metrics']['test_f1'])
            self.best_model_name = best_model_name

            logger.info(
                f"Best model: {best_model_name} with f1 score: {all_results[best_model_name]['metrics']['test_f1']:.4f}")
        return all_results

    def save_model(self, model_name: str, version: str = None) -> str:
        """
        Save trained model and associated components to disk.

        :param model_name: Name of the model to save.
        :param version: Model version (auto-generated if None).
        :return: Path where the model was saved.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train the model first.")

        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model = self.models[model_name]
        scaler = self.scalers[model_name]

        # Save model
        model_file = self.save_path / f"{model_name}_model_{version}.pkl"
        joblib.dump(model, model_file)

        # Save scaler
        scaler_file = self.save_path / f"{model_name}_scaler_{version}.pkl"
        joblib.dump(scaler, scaler_file)

        # Save metadata
        metadata = {
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'class_names': self.class_names,
            'model_type': str(type(model).__name__),
        }

        # Add performance metrics if available
        for result in self.training_history:
            if result['model_name'] == model_name:
                metadata.update({
                    'performance': result['metrics'],
                    'cv_scores': result['cv_scores'],
                    'training_time': result['training_time']
                })
                break

        metadata_file = self.save_path / f"{model_name}_metadata_{version}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model {model_name} saved:")
        logger.info(f"  Model: {model_file}")
        logger.info(f"  Scaler: {scaler_file}")
        logger.info(f"  Metadata: {metadata_file}")

        return str(model_file)

    def save_best_model(self, version: str = None) -> Optional[str]:
        """
        Save the best performing model.

        :param version: Model version (auto-generated if None).
        :return: Path where the best model was saved, or None if no models trained.
        """
        if self.best_model_name is None:
            logger.warning("No best model identified. Train models first.")
            return None

        return self.save_model(model_name=self.best_model_name, version=version)

    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.

        :param model_name: Name of the model to evaluate.
        :param X_test: Test features.
        :param y_test: Test targets.
        :return: Dictionary with evaluation results.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        scaler = self.scalers[model_name]

        # Scale test features
        X_test_scaled = scaler.transform(X_test)

        # Generate predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate detailed metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_test, y_pred)

        # Add confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)

        evaluation_results = {
            'model_name': model_name,
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat()
        }

        return evaluation_results


    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all training sessions.

        :return: Dictionary with training history and model comparisons.
        """
        if not self.training_history:
            return {'message': 'No models trained yet'}

        # Model comparison
        model_comparison = []
        for result in self.training_history:
            model_comparison.append({
                'model_name': result['model_name'],
                'test_accuracy': result['metrics']['test_accuracy'],
                'test_f1': result['metrics']['test_f1'],
                'cv_accuracy_mean': result['cv_scores']['mean'],
                'cv_accuracy_std': result['cv_scores']['std'],
                'training_time': result['training_time'],
                'timestamp': result['timestamp']
            })

        # Sort by test accuracy
        model_comparison.sort(key=lambda x: x['test_accuracy'], reverse=True)

        summary = {
            'total_models_trained': len(self.training_history),
            'best_model': self.best_model_name,
            'model_comparison': model_comparison,
            'training_history': self.training_history,
            'available_models': list(self.models.keys()),
        }

        return summary

    def load_model(self, model_path: str) -> Tuple[Any, StandardScaler, Dict[str, Any]]:
        """
        Load a previously saved model.

        :param model_path: Path to the saved model file.
        :return: Tuple of (model, scaler, metadata).
        """
        model_path = Path(model_path)

        # Load model
        model = joblib.load(model_path)

        # Try to load scaler and metadata
        base_name = model_path.stem.replace('_model_', '_')
        scaler_path = model_path.parent / f"{base_name.replace('_', '_scaler_', 1)}.pkl"
        metadata_path = model_path.parent / f"{base_name.replace('_', '_metadata_', 1)}.json"

        scaler = None
        metadata = {}

        if scaler_path.exists():
            scaler = joblib.load(scaler_path)

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        logger.info(f"Loaded model from {model_path}")
        return model, scaler, metadata

    def _get_model_configurations(self) -> Dict[str, Dict[str, Any]]:
        """
        Define configurations for different ML models.

        :return: Dictionary with model configurations.
        """
        configs = {
            'SGDClassifier': {
                'model': SGDClassifier(random_state=self.random_state),
                'param_grid': {
                    'loss': ['hinge', 'log_loss', 'modified_huber'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha': [1e-4, 1e-3, 1e-2],
                    'max_iter': [1000, 5000],
                    'class_weight': [None, 'balanced']
                },
                'handle_imbalance': True
            },
            'LinearSVC': {
                #'model': LinearSVC(random_state=self.random_state, class_weight='balanced', max_iter=10000),
                'model': LinearSVC(random_state=self.random_state),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],  # Only 'l2' allowed in LinearSVC
                    'loss': ['squared_hinge'],
                    'dual': [True, False]  # Only certain combinations valid
                },
                'handle_imbalance': True
            },
            'RidgeClassifier': {
                #'model': RidgeClassifier(random_state=self.random_state, class_weight='balanced'),
                'model': RidgeClassifier(random_state=self.random_state),
                'param_grid': {
                    'alpha': [0.1, 1.0, 10.0],
                    'solver': ['auto', 'sag', 'lsqr', 'sparse_cg']
                },
                'handle_imbalance': True
            },
            'RidgeClassifierCV': {
                #'model': RidgeClassifierCV(class_weight='balanced'),
                'model': RidgeClassifierCV(),
                'param_grid': {
                    'alphas': [0.1, 1.0, 10.0],  # Must be a list of lists
                    'store_cv_values': [True, False]
                },
                'handle_imbalance': True
            },
            'GaussianNB': {
                'model': GaussianNB(),
                'param_grid': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7]
                },
                'handle_imbalance': False  # doesn't support class_weight
            },
            'NearestCentroid': {
                'model': NearestCentroid(),
                'param_grid': {
                    'metric': ['euclidean', 'manhattan'],
                    'shrink_threshold': [None, 0.1, 0.5]
                },
                'handle_imbalance': False  # doesn't support class_weight
            },
            'QuadraticDiscriminantAnalysis': {
                'model': QuadraticDiscriminantAnalysis(),
                'param_grid': {
                    'reg_param': [0.0, 0.1, 0.3],
                    'store_covariance': [True, False]
                },
                'handle_imbalance': False  # QDA also doesn't support class_weight directly
            },
            'LGBMClassifier': {
                'model': LGBMClassifier(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'num_leaves': [31, 63], #[?]todo, try [15, 31, 63, 127]
                    'learning_rate': [0.01, 0.1, 0.2],
                    'boosting_type': ['gbdt', 'dart'],
                    'class_weight': ['balanced', None]
                },
                'handle_imbalance': True
            },
            'ExtraTreesClassifier': {
                #'model': ExtraTreesClassifier(random_state=self.random_state, n_jobs=1, class_weight='balanced'),
                'model': ExtraTreesClassifier(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'handle_imbalance': True
            },
            'NuSVC': {
                #'model': NuSVC(random_state=self.random_state, probability=True, class_weight='balanced'),
                'model': NuSVC(random_state=self.random_state),
                'param_grid': {
                    'nu': [0.3, 0.5, 0.7],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'handle_imbalance': True
            },
            'SGDClassifier': {
                #'model': SGDClassifier(random_state=self.random_state, class_weight='balanced', max_iter=1000),
                'model': SGDClassifier(random_state=self.random_state),
                'param_grid': {
                    'loss': ['hinge', 'log_loss', 'modified_huber'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha': [1e-4, 1e-3, 1e-2]
                },
                'handle_imbalance': True
            },
            'LinearDiscriminantAnalysis': {
                'model': LinearDiscriminantAnalysis(),
                'param_grid': {
                    'solver': ['svd', 'lsqr', 'eigen'],
                    'shrinkage': [None, 'auto']  # Only used with lsqr and eigen
                },
                'handle_imbalance': False  # LDA doesn't support class_weight directly
            },
            'RandomForestClassifier': {
                #'model': RandomForestClassifier(random_state=self.random_state, n_jobs=1),
                'model': RandomForestClassifier(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'handle_imbalance': True
            },
            'XGBClassifier': {
                #'model': xgb.XGBClassifier(random_state=self.random_state, eval_metric='mlogloss', n_jobs=1),
                'model': xgb.XGBClassifier(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'handle_imbalance': True
            },
            'LogisticRegression': {
                #'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'model': LogisticRegression(random_state=self.random_state),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'handle_imbalance': True
            },
            'SVC': {
                #'model': SVC(random_state=self.random_state, probability=True),
                'model': SVC(random_state=self.random_state),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'handle_imbalance': True
            }
        }

        return configs

    def _calculate_metrics(self, y_train: np.ndarray, y_train_pred: np.ndarray,
                           y_test: np.ndarray, y_test_pred: np.ndarray,
                           ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        :param y_train: True training labels.
        :param y_train_pred: Predicted training labels.
        :param y_test: True test labels.
        :param y_test_pred: Predicted test labels.
        :return: Dictionary with various metrics.
        """
        metrics = {
            # Accuracy metrics
            'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
            'test_accuracy': float(accuracy_score(y_test, y_test_pred)),

            # Precision, Recall, F1 (macro average)
            'test_precision': float(precision_score(y_test, y_test_pred, average='macro')),
            'test_recall': float(recall_score(y_test, y_test_pred, average='macro')),
            'test_f1': float(f1_score(y_test, y_test_pred, average='macro')),

            # Class-specific metrics
            'test_precision_weighted': float(precision_score(y_test, y_test_pred, average='weighted')),
            'test_recall_weighted': float(recall_score(y_test, y_test_pred, average='weighted')),
            'test_f1_weighted': float(f1_score(y_test, y_test_pred, average='weighted'))
        }

        return metrics

    def create_ensemble_model(self, model_names: List[str], method: str = 'voting') -> Dict[str, Any]:
        """
        Create ensemble model from multiple trained models.

        :param model_names: List of model names to combine.
        :param method: Ensemble method ('voting' or 'stacking').
        :return: Dictionary with ensemble model results.
        """
        if method == 'voting':
            from sklearn.ensemble import VotingClassifier

            # Prepare models for voting
            estimators = []
            for name in model_names:
                if name in self.models:
                    estimators.append((name, self.models[name]))

            if len(estimators) < 2:
                raise ValueError("Need at least 2 models for ensemble")

            # Create voting classifier
            ensemble = VotingClassifier(estimators=estimators, voting='soft')

            # Note: This is a simplified implementation
            # In practice, you'd need to retrain on the same data
            logger.info(f"Created ensemble model with {len(estimators)} models")

            return {
                'ensemble_model': ensemble,
                'component_models': model_names,
                'method': method
            }

        else:
            raise NotImplementedError(f"Ensemble method '{method}' not implemented")


def train():
    before = datetime.now()
    trainer = FootballModelTrainer()
    #todo: add scaling, imputer, get_processed_features(data_df: pd.DataFrame, tasks=['scaling', 'imputation'])
    X_train, X_test, y_train, y_test = trainer.prepare_training_data()
    # [?]todo: Add fine tuning, for instant after fine tuning doesn't improve f1 score
    results = trainer.train_top_models(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, hyperparameter_tuning=False
    )
    trainer.save_best_model()
    logger.info(f'Execution time =  {datetime.now() - before}')


if __name__ == '__main__':
    train()
