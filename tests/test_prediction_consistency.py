"""Test prediction consistency for FootballPredictor."""

import pytest
from models.predictor import FootballPredictor


class TestPredictionConsistency:
    """Test that predictions are deterministic and consistent."""

    def test_multiple_predictions_same_result(self):
        """Test that calling predict multiple times returns identical results."""
        predictor = FootballPredictor()
        
        # Make 5 predictions with same inputs
        results = []
        for _ in range(5):
            result = predictor.predict(
                league_id=88,
                team_a_name="Fortuna Sittard", 
                team_b_name="GO Ahead Eagles"
            )
            results.append(result)
        
        # All predictions should be identical
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            # Handle case where prediction fails (model not trained)
            if not result.get('success'):
                assert result.get('error') == first_result.get('error'), \
                    f"Prediction {i+1} error differs from first prediction"
                continue
                
            assert result['prediction']['outcome'] == first_result['prediction']['outcome'], \
                f"Prediction {i+1} outcome differs from first prediction"
            assert result['prediction']['confidence'] == first_result['prediction']['confidence'], \
                f"Prediction {i+1} confidence differs from first prediction"
            assert result['prediction']['probabilities'] == first_result['prediction']['probabilities'], \
                f"Prediction {i+1} probabilities differ from first prediction"

    def test_different_matches_different_results(self):
        """Test that different match inputs produce different results."""
        predictor = FootballPredictor()
        
        # Predict two different matches
        result1 = predictor.predict(88, "Ajax", "PSV")
        result2 = predictor.predict(88, "Feyenoord", "AZ Alkmaar")
        
        # Results should be different (at least probabilities should differ)
        assert result1['prediction']['probabilities'] != result2['prediction']['probabilities'], \
            "Different matches should produce different prediction probabilities"

    def test_new_predictor_instances_consistency(self):
        """Test that new predictor instances return same results."""
        # Create separate predictor instances
        predictor1 = FootballPredictor()
        predictor2 = FootballPredictor()
        
        result1 = predictor1.predict(88, "Fortuna Sittard", "GO Ahead Eagles")
        result2 = predictor2.predict(88, "Fortuna Sittard", "GO Ahead Eagles")
        
        # Should be identical across instances
        assert result1['prediction']['outcome'] == result2['prediction']['outcome']
        assert result1['prediction']['confidence'] == result2['prediction']['confidence']
        assert result1['prediction']['probabilities'] == result2['prediction']['probabilities']

    def test_rapid_successive_predictions(self):
        """Test rapid successive predictions for consistency."""
        predictor = FootballPredictor()
        
        results = []
        for i in range(10):
            result = predictor.predict(88, "Fortuna Sittard", "GO Ahead Eagles")
            results.append((i, result['prediction']['confidence'], result['prediction']['outcome']))
        
        # Print all results to debug
        for i, confidence, outcome in results:
            print(f"Prediction {i}: {outcome} - {confidence}")
        
        # All should be identical
        first_confidence = results[0][1]
        first_outcome = results[0][2]
        
        for i, confidence, outcome in results:
            assert outcome == first_outcome, f"Prediction {i} outcome changed: {outcome} != {first_outcome}"
            assert confidence == first_confidence, f"Prediction {i} confidence changed: {confidence} != {first_confidence}"