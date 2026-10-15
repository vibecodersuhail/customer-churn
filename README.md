# Customer Churn Prediction (PBADT304 Introduction to Data Science)

FastAPI backend + HTML/JS frontend that walks through the full data science process:
Business Objective → Data Requirement → Data Collection → Data Preparation → EDA →
Modelling → Evaluation → Deployment → Monitoring.

## Run
    python -m pip install -r requirements.txt
    python main.py
Open http://127.0.0.1:8000 (or double-click run_app.bat).

## Structure
    main.py            backend: data cleaning, EDA, 3 models, REST API
    static/index.html  frontend: opening animation, 9 tabs, 19 charts
    static/chart.umd.min.js  Chart.js (offline)
    requirements.txt
    run_app.bat
Optional: put a churn CSV in this folder (or ./data) or upload it on the Data Collection tab.

## API
GET /api/overview /preparation /eda /models /schema /monitoring /log /cleaned.csv
POST /api/predict /feedback /retrain /upload
