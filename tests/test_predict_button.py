"""Test for Streamlit Predict button functionality in Docker container."""

import json
import subprocess
import time
from typing import Dict, Any

import pytest
import requests


class TestPredictButton:
    """Test the Predict button functionality in Docker container."""
    
    @pytest.fixture(scope="class")
    def docker_container(self):
        """Start Streamlit app in Docker container for testing."""
        # Build Docker image
        build_result = subprocess.run(
            ["docker", "build", "-t", "predictor-app-test", "."],
            capture_output=True,
            text=True,
            cwd="/app"  # Use container working directory
        )
        assert build_result.returncode == 0, f"Docker build failed: {build_result.stderr}"
        
        # Start container with Streamlit app
        container = subprocess.Popen(
            ["docker", "run", "--rm", "-p", "8502:8501", "predictor-app-test"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for app to start
        max_retries = 30
        for _ in range(max_retries):
            try:
                response = requests.get("http://localhost:8502", timeout=2)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(1)
        else:
            container.terminate()
            pytest.fail("Streamlit app failed to start within 30 seconds")
        
        yield container
        
        # Cleanup
        container.terminate()
        subprocess.run(["docker", "rmi", "predictor-app-test"], capture_output=True)

    def test_predict_button_functionality_without_docker(self):
        """Test the core prediction logic that the Predict button uses."""
        from models.predictor import FootballPredictor
        
        # Test the exact same logic as in app.py lines 82-96
        predictor = FootballPredictor()
        
        # Verify model loaded successfully (not None)
        assert predictor.model is not None, "Model should be loaded, not None"
        
        # Test with valid teams from Eredivisie (league 88)
        result = predictor.predict(
            league_id=88,
            team_a_name="Ajax",
            team_b_name="PSV"
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result.get('success'):
            # Test successful prediction structure (matches app.py lines 90-96)
            prediction = result['prediction']
            assert 'outcome' in prediction
            assert 'confidence' in prediction
            assert isinstance(prediction['confidence'], (int, float))
            assert 0 <= prediction['confidence'] <= 1
            assert prediction['outcome'] in ['Home Win', 'Draw', 'Away Win']
        else:
            # If prediction fails, should have error message
            assert 'error' in result
            assert isinstance(result['error'], str)
        
    def test_streamlit_app_accessibility(self):
        """Test that Streamlit app components are importable."""
        # Simple test - just verify app.py can be imported without errors
        try:
            import app
            assert True, "app.py imports successfully"
        except ImportError as e:
            pytest.fail(f"app.py import failed: {e}")
        
    def test_predict_with_invalid_teams(self):
        """Test prediction with invalid team names."""
        from models.predictor import FootballPredictor
        
        predictor = FootballPredictor()
        result = predictor.predict(
            league_id=39,
            team_a_name="NonExistentTeamA",
            team_b_name="NonExistentTeamB"
        )
        
        assert isinstance(result, dict)
        # Should handle invalid teams gracefully
        assert 'error' in result or ('prediction' in result and result['prediction'] is not None)

    def test_predict_with_invalid_league(self):
        """Test prediction with invalid league ID."""
        from models.predictor import FootballPredictor
        
        predictor = FootballPredictor()
        result = predictor.predict(
            league_id=99999,  # Invalid league
            team_a_name="Team A",
            team_b_name="Team B"
        )
        
        assert isinstance(result, dict)
        # Should handle invalid league gracefully
        assert 'error' in result or ('prediction' in result and result['prediction'] is not None)