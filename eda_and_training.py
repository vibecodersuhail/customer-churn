"""
eda_and_training.py
-------------------
Customer Churn Prediction Using Data Science and Machine Learning

This script performs:
1. Data Loading: Loads the Telco Customer Churn dataset.
2. Data Cleaning & Preprocessing: Handles missing values, data types, and target encoding.
3. Exploratory Data Analysis (EDA): Generates and saves visual plots to the 'figures/' directory.
4. Train-Test Split: Splits data into 80% training and 20% testing sets (stratified).
5. Model Training: Fits a Logistic Regression classifier inside a Scikit-Learn Pipeline.
6. Model Evaluation: Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
7. Model Serialization: Saves the trained pipeline to 'models/churn_pipeline.joblib'.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Seaborn styling for clear, aesthetic plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def load_data(filepath: str) -> pd.DataFrame:
    """Loads CSV dataset and reports initial dimensions."""
    print("=" * 60)
    print("STEP 1: LOADING DATASET")
    print("=" * 60)
    if not os.path.exists(filepath):
        from data.download_data import ensure_dataset
        ensure_dataset()

    df = pd.read_csv(filepath)
    print(f"Dataset successfully loaded from: {filepath}")
    print(f"Total Rows (Customers): {df.shape[0]}")
    print(f"Total Columns (Features): {df.shape[1]}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data:
    - Converts TotalCharges to numeric (handles whitespace strings).
    - Imputes missing TotalCharges for customers with 0 tenure.
    - Encodes Churn target ('Yes' -> 1, 'No' -> 0).
    - Drops customerID as it is an arbitrary identifier.
    """
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING & PREPROCESSING")
    print("=" * 60)
    df_clean = df.copy()

    # 1. TotalCharges conversion: spaces become NaN
    initial_blanks = (df_clean["TotalCharges"].str.strip() == "").sum()
    print(f"Detected {initial_blanks} blank string entries in 'TotalCharges'.")
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    
    # Fill NaN with 0 for tenure=0 customers
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)
    print("Replaced blank 'TotalCharges' with 0.0 for new customers with 0 tenure.")

    # 2. Drop customerID
    if "customerID" in df_clean.columns:
        df_clean.drop(columns=["customerID"], inplace=True)
        print("Dropped non-predictive column 'customerID'.")

    # 3. Encode target Churn: Yes -> 1, No -> 0
    df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0})
    churn_counts = df_clean["Churn"].value_counts()
    print(f"Target distribution (Churn):")
    print(f"  - No  (Retained): {churn_counts[0]} ({churn_counts[0] / len(df_clean):.1%})")
    print(f"  - Yes (Churned) : {churn_counts[1]} ({churn_counts[1] / len(df_clean):.1%})")

    return df_clean


def generate_eda(df: pd.DataFrame):
    """
    Generates and saves exploratory graphs:
    1. Overall Churn Distribution
    2. Churn by Contract Type
    3. Monthly Charges Distribution by Churn
    4. Customer Tenure Distribution by Churn
    """
    print("\n" + "=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # 1. Customer Churn Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    churn_labels = ["Retained (No)", "Churned (Yes)"]
    counts = df["Churn"].value_counts().sort_index()

    sns.barplot(x=churn_labels, y=counts.values, hue=churn_labels, ax=axes[0], palette=["#2ecc71", "#e74c3c"], legend=False)
    axes[0].set_title("Customer Churn Count", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Customers")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, f"{v:,}", ha="center", fontweight="bold")

    axes[1].pie(
        counts.values,
        labels=churn_labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ecc71", "#e74c3c"],
        explode=(0, 0.07),
        wedgeprops=dict(width=0.6, edgecolor="white", linewidth=2),
    )
    axes[1].set_title("Customer Churn Percentage", fontsize=13, fontweight="bold")

    plot1_path = os.path.join(FIGURES_DIR, "churn_distribution.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved EDA 1] {plot1_path}")

    # 2. Churn by Contract Type
    plt.figure(figsize=(9, 5))
    contract_df = df.copy()
    contract_df["Churn_Label"] = contract_df["Churn"].map({1: "Churned", 0: "Retained"})
    ax = sns.countplot(
        data=contract_df,
        x="Contract",
        hue="Churn_Label",
        palette={"Retained": "#2ecc71", "Churned": "#e74c3c"}
    )
    plt.title("Customer Churn by Contract Type", fontsize=14, fontweight="bold")
    plt.xlabel("Contract Type", fontsize=12)
    plt.ylabel("Number of Customers", fontsize=12)
    plt.legend(title="Status", frameon=True)
    
    # Annotate bar heights
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 3),
                textcoords="offset points"
            )

    plot2_path = os.path.join(FIGURES_DIR, "contract_churn.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved EDA 2] {plot2_path}")

    # 3. Monthly Charges by Churn
    plt.figure(figsize=(9, 5))
    sns.kdeplot(
        data=df[df["Churn"] == 0]["MonthlyCharges"],
        label="Retained (No Churn)",
        color="#2ecc71",
        fill=True,
        alpha=0.4,
        linewidth=2
    )
    sns.kdeplot(
        data=df[df["Churn"] == 1]["MonthlyCharges"],
        label="Churned",
        color="#e74c3c",
        fill=True,
        alpha=0.4,
        linewidth=2
    )
    plt.title("Monthly Charges Distribution by Churn Status", fontsize=14, fontweight="bold")
    plt.xlabel("Monthly Charges ($)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(frameon=True)

    plot3_path = os.path.join(FIGURES_DIR, "monthly_charges_churn.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved EDA 3] {plot3_path}")

    # 4. Tenure by Churn
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=contract_df,
        x="tenure",
        hue="Churn_Label",
        multiple="stack",
        bins=30,
        palette={"Retained": "#2ecc71", "Churned": "#e74c3c"}
    )
    plt.title("Customer Tenure Distribution (Months with Company)", fontsize=14, fontweight="bold")
    plt.xlabel("Tenure (Months)", fontsize=12)
    plt.ylabel("Customer Count", fontsize=12)

    plot4_path = os.path.join(FIGURES_DIR, "tenure_churn.png")
    plt.savefig(plot4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved EDA 4] {plot4_path}")


def train_and_evaluate(df: pd.DataFrame):
    """
    Splits data, builds preprocessing pipeline with Logistic Regression,
    evaluates on test set, and exports model artifact.
    """
    print("\n" + "=" * 60)
    print("STEP 4: TRAIN-TEST SPLIT")
    print("=" * 60)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Identify numerical and categorical columns
    numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [col for col in X.columns if col not in numerical_features]

    print(f"Features: {X.shape[1]} total ({len(numerical_features)} numerical, {len(categorical_features)} categorical)")
    print(f"Numerical columns  : {numerical_features}")
    print(f"Categorical columns: {categorical_features}")

    # 80/20 train-test split with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size : {X_test.shape[0]} samples")

    print("\n" + "=" * 60)
    print("STEP 5: BUILDING PIPELINE & TRAINING LOGISTIC REGRESSION")
    print("=" * 60)

    # Preprocessing transformers
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Full pipeline with Logistic Regression
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    print("Fitting model on training set...")
    pipeline.fit(X_train, y_train)
    print("[SUCCESS] Model training complete.")

    print("\n" + "=" * 60)
    print("STEP 6: MODEL EVALUATION ON UNSEEN TEST DATA")
    print("=" * 60)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy  : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision : {prec:.4f} ({prec * 100:.2f}%)")
    print(f"Recall    : {rec:.4f} ({rec * 100:.2f}%)")
    print(f"F1-Score  : {f1:.4f} ({f1 * 100:.2f}%)")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained (0)", "Churned (1)"]))

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Stay", "Predicted Churn"],
        yticklabels=["Actual Stay", "Actual Churn"],
        cbar=False
    )
    plt.title("Confusion Matrix - Logistic Regression", fontsize=13, fontweight="bold")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")

    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved Evaluation] {cm_path}")

    # Top Feature Influences
    ohe_cols = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
        .tolist()
    )
    all_feature_names = numerical_features + ohe_cols
    coefficients = pipeline.named_steps["classifier"].coef_[0]

    coef_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Coefficient": coefficients,
        "AbsCoeff": np.abs(coefficients)
    }).sort_values(by="AbsCoeff", ascending=False)

    print("\nTop 5 Drivers Increasing Churn Risk (Positive Coefficients):")
    top_churn = coef_df[coef_df["Coefficient"] > 0].head(5)
    for _, row in top_churn.iterrows():
        print(f"  + {row['Feature']}: {row['Coefficient']:.3f}")

    print("\nTop 5 Drivers Increasing Customer Retention (Negative Coefficients):")
    top_stay = coef_df[coef_df["Coefficient"] < 0].head(5)
    for _, row in top_stay.iterrows():
        print(f"  - {row['Feature']}: {row['Coefficient']:.3f}")

    # Save Pipeline and Metadata
    model_path = os.path.join(MODELS_DIR, "churn_pipeline.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\n[Saved Model Pipeline] {model_path}")

    metadata = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "categorical_options": {col: sorted(df[col].unique().tolist()) for col in categorical_features}
    }

    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Saved Metadata] {meta_path}")

    return pipeline, metadata


def main():
    df = load_data(DATA_PATH)
    df_clean = clean_data(df)
    generate_eda(df_clean)
    train_and_evaluate(df_clean)
    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
