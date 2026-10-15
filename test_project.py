"""
test_project.py
---------------
Automated test suite to verify project integrity:
1. Dataset existence and non-emptiness.
2. Saved model pipeline loading and inference verification.
3. Metadata validation.
4. Generated EDA and evaluation figures existence.
"""

import os
import json
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.joblib")
META_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


def test_dataset_exists():
    assert os.path.exists(DATA_PATH), f"Dataset file missing at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    assert len(df) == 7043, f"Expected 7,043 rows, found {len(df)}"
    assert "Churn" in df.columns, "Target column 'Churn' missing from dataset"
    print("[PASS] Dataset integrity test passed.")


def test_figures_exist():
    expected_figures = [
        "churn_distribution.png",
        "contract_churn.png",
        "monthly_charges_churn.png",
        "tenure_churn.png",
        "confusion_matrix.png"
    ]
    for fig_name in expected_figures:
        fig_path = os.path.join(FIGURES_DIR, fig_name)
        assert os.path.exists(fig_path), f"Figure missing: {fig_path}"
        assert os.path.getsize(fig_path) > 1000, f"Figure file seems empty: {fig_path}"
    print("[PASS] Figures verification passed.")


def test_model_and_metadata():
    assert os.path.exists(MODEL_PATH), f"Model pipeline missing at {MODEL_PATH}"
    assert os.path.exists(META_PATH), f"Metadata missing at {META_PATH}"

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    assert meta["accuracy"] >= 0.75, f"Model accuracy too low: {meta['accuracy']}"
    assert "numerical_features" in meta
    assert "categorical_features" in meta
    print(f"[PASS] Metadata verification passed (Accuracy: {meta['accuracy'] * 100:.2f}%).")


def test_model_inference():
    pipeline = joblib.load(MODEL_PATH)

    # Test sample with valid schema
    sample_payload = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0
    }])

    pred = pipeline.predict(sample_payload)
    prob = pipeline.predict_proba(sample_payload)

    assert pred[0] in [0, 1], f"Prediction must be 0 or 1, got {pred[0]}"
    assert prob.shape == (1, 2), f"Probabilities shape must be (1, 2), got {prob.shape}"
    assert round(float(prob[0].sum()), 2) == 1.0, "Probabilities must sum to 1.0"
    print(f"[PASS] Model inference test passed (Prediction: {pred[0]}, Probability: {prob[0]}).")


if __name__ == "__main__":
    test_dataset_exists()
    test_figures_exist()
    test_model_and_metadata()
    test_model_inference()
    print("\nALL AUTOMATED TESTS PASSED SUCCESSFULLY! (4/4)")
