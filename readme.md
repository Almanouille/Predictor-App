# 📦 Predictor-App

### 📚 Project Structure

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
│   ├── test_models.py 
│   └── test_etl.py 
├── etl.py                   # Data extraction, transform, and load 
├── requirements.txt 
├── Dockerfile               # To support ETL, training, and Streamlit UI
├── docker-compose.yml       # For database + app
├── .env.example             # Environment variables template
└── README.md
```
### Run Steps
#### **Part I**:Setup Instructions
* Create a virtual environment using Python 3.8: `python3.8 -m venv venv`
* Activate the virtual environment: `source venv/bin/activate`
* Install dependencies: `pip install -r requirements.txt`

#### **Part II**: Load history match data
* Run `python etl.py`, it will store data in raw and processed folders

#### **Part III**: Modeling
* Train and save model: run `python -m models.trainer` 
* Update `SELECTED_FEATURES` value in config.py 
* Predict: `python -m models.predictor`

#### **Part IV**: Predict on an application streamlit
* Run: `streamlit run app.py`