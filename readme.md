# Predictor-App

### Project Structure

```text
project-root/
├── app.py                     # Streamlit main app
├── models/
│   ├── __init__.py
│   ├── predictor.py          # Model class 
│   ├── trainer.py            # Training logic
│   ├── feature_calculator.py # feature computation logic
│   └── saved_models/        # Trained model files 
├── data/                            # Raw and processed data (from your ETL)
│   ├── raw/                         # Raw JSON data from API
│   │   ├── fixtures_39_2023.json    # Example for league_id 39 season 2023
│   │   ├── standings_39_2023.json
│   │   └── team_stats_1_39_2023.json
│   ├── processed/                   # Cleaned CSV data (ML training ready)
│   │   ├── fixtures_cleaned_39_2023.csv
│   │   ├── standings_cleaned_39_2023.csv
│   │   └── team_stats_cleaned_39_2023.csv
│   └── consolidated/                # Final ML-ready datasets
│       ├── fixtures_consolidated_20240327_143022.csv
│       └── team_statistics_consolidated_20240327_143022.csv
├── config.py                # Handles env vars
├── utils/
│   ├── __init__.py 
│   └── helpers.py            # Utility functions 
├── tests/ 
│   ├── test_predict_button.py        # Streamlit predict button tests
│   └── test_prediction_consistency.py # Model consistency tests   
├── etl.py                   # Data extraction, transform, and load 
├── requirements.txt 
├── Dockerfile               # To support ETL, training, and Streamlit UI
├── docker-compose.yml       # For database + app
├── docker-compose.dev.yml   # Development mode with volume mounts
├── .env.example             # Environment variables template
├── CLAUDE.md                # AI assistant context and commands
└── README.md
```
### Run Steps
#### **Part I**: Install Docker
* Mac: `brew install --cask docker`
* Linux: follow steps on https://docs.sevenbridges.com/docs/install-docker-on-linux 

#### **Part II**: Build docker image named app : `docker build -t <image> .` (check image list `docker images`)

#### **Part III**: Run docker image in container 
* Run the Streamlit UI: run `docker run -p 8501:8501 <image>` 
* Run ETL process to get API data: run `docker run <image> python etl.py`
* Run model training: 
  * Run `docker run <image> python -m models.trainer`
  * Update `SELECTED_FEATURES` value in logs in config.py
* Run prediction without UI: run `docker run <image> python -m models.predictor` 
* Run all tests: run `docker run <image> python -m pytest tests/ -v`

#### **Part IV**: Development mode (avoid rebuilds)
Use volume mounts to sync code changes without rebuilding:
$(pwd) ensures you get the full absolute path regardless of which directory you're in
* Run tests: `docker run -v $(pwd):/app <image> python -m pytest tests/ -v`
* Run training: `docker run -v $(pwd):/app <image> python -m models.trainer`
* Run Streamlit: `docker run -v $(pwd):/app -p 8501:8501 <image>`
* Or use docker-compose: `docker-compose -f docker-compose.dev.yml run tests`

#### **Clean up**: remove all Docker images`docker rmi -f $(docker images -aq)`