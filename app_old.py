"""
app.py
------
Streamlit Web Application: Customer Churn Prediction Using Data Science and Machine Learning
Designed for BTech Data Science Students & Pair Programming Evaluation.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction | ML App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, student-friendly presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
        margin-top: 4px;
    }
    .churn-alert-danger {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 16px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .churn-alert-safe {
        background-color: #DCFCE7;
        border-left: 5px solid #22C55E;
        padding: 16px;
        border-radius: 8px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Paths & Artifact Loading
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.joblib")
META_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


@st.cache_resource
def load_model_and_meta():
    """Loads the trained machine learning pipeline and metadata."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        st.error("Model files not found! Please run `python eda_and_training.py` first.")
        st.stop()
    pipeline = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return pipeline, meta


pipeline, metadata = load_model_and_meta()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/clouds/200/graph.png", width=110)
st.sidebar.title("Project Navigator")
st.sidebar.markdown("**Project:** Customer Churn Prediction")
st.sidebar.markdown("**Algorithm:** Logistic Regression")
st.sidebar.markdown(f"**Model Accuracy:** `{metadata['accuracy'] * 100:.1f}%`")
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Demo Presets")
st.sidebar.markdown("Select a profile to quickly test the model:")

preset_choice = st.sidebar.radio(
    "Choose Preset:",
    ("Custom Inputs", "High-Risk Customer (Likely to Churn)", "Loyal Customer (Likely to Stay)")
)

# ---------------------------------------------------------
# Preset Configurations
# ---------------------------------------------------------
if preset_choice == "High-Risk Customer (Likely to Churn)":
    default_gender = "Female"
    default_senior = 0
    default_partner = "No"
    default_dependents = "No"
    default_tenure = 2
    default_phone = "Yes"
    default_multlines = "No"
    default_internet = "Fiber optic"
    default_security = "No"
    default_backup = "No"
    default_protection = "No"
    default_support = "No"
    default_tv = "Yes"
    default_movies = "Yes"
    default_contract = "Month-to-month"
    default_paperless = "Yes"
    default_payment = "Electronic check"
    default_monthly = 89.85
    default_total = 179.70
elif preset_choice == "Loyal Customer (Likely to Stay)":
    default_gender = "Male"
    default_senior = 0
    default_partner = "Yes"
    default_dependents = "Yes"
    default_tenure = 60
    default_phone = "Yes"
    default_multlines = "Yes"
    default_internet = "DSL"
    default_security = "Yes"
    default_backup = "Yes"
    default_protection = "Yes"
    default_support = "Yes"
    default_tv = "No"
    default_movies = "No"
    default_contract = "Two year"
    default_paperless = "No"
    default_payment = "Credit card (automatic)"
    default_monthly = 55.20
    default_total = 3312.00
else:
    default_gender = "Male"
    default_senior = 0
    default_partner = "No"
    default_dependents = "No"
    default_tenure = 12
    default_phone = "Yes"
    default_multlines = "No"
    default_internet = "DSL"
    default_security = "No"
    default_backup = "Yes"
    default_protection = "No"
    default_support = "No"
    default_tv = "No"
    default_movies = "No"
    default_contract = "Month-to-month"
    default_paperless = "Yes"
    default_payment = "Mailed check"
    default_monthly = 53.85
    default_total = 646.20

# ---------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------
st.markdown("<div class='main-title'>Customer Churn Prediction System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>An End-to-End Data Science & Machine Learning Application "
    "built with Python, Scikit-Learn, and Streamlit.</div>",
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Churn Predictor",
    "📈 Exploratory Data Analysis (EDA)",
    "🧠 Model Evaluation & Metrics",
    "📚 BTech Student Guide & Dataset"
])

# =========================================================
# TAB 1: CHURN PREDICTOR
# =========================================================
with tab1:
    st.markdown("### Enter Customer Details for Prediction")
    st.markdown("Fill in the customer information below, then click **Predict Churn Risk**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Customer Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"], index=0 if default_gender == "Female" else 1)
        senior = st.selectbox(
            "Senior Citizen (Age 65+)",
            [0, 1],
            index=default_senior,
            format_func=lambda x: "Yes (Senior)" if x == 1 else "No"
        )
        partner = st.selectbox("Has Partner / Spouse?", ["Yes", "No"], index=0 if default_partner == "Yes" else 1)
        dependents = st.selectbox("Has Dependents / Children?", ["Yes", "No"], index=0 if default_dependents == "Yes" else 1)

    with col2:
        st.subheader("📋 Account & Contract")
        contract = st.selectbox(
            "Contract Term",
            ["Month-to-month", "One year", "Two year"],
            index=["Month-to-month", "One year", "Two year"].index(default_contract)
        )
        tenure = st.slider("Customer Tenure (Months with Company)", min_value=0, max_value=72, value=int(default_tenure), step=1)
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if default_paperless == "Yes" else 1)
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            index=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ].index(default_payment)
        )

    with col3:
        st.subheader("💰 Charges")
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=15.0,
            max_value=150.0,
            value=float(default_monthly),
            step=1.0
        )
        # Suggest total charges automatically if desired
        calc_total = round(tenure * monthly_charges, 2)
        total_charges = st.number_input(
            "Total Charges ($) [Estimated: tenure × monthly]",
            min_value=0.0,
            max_value=10000.0,
            value=float(default_total if default_total > 0 else calc_total),
            step=10.0
        )

    st.markdown("---")
    st.subheader("🌐 Telecom & Internet Services Subscribed")
    sc1, sc2, sc3, sc4 = st.columns(4)

    with sc1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if default_phone == "Yes" else 1)
        multiple_lines = st.selectbox(
            "Multiple Phone Lines",
            ["No", "Yes", "No phone service"],
            index=["No", "Yes", "No phone service"].index(default_multlines)
        )

    with sc2:
        internet_service = st.selectbox(
            "Internet Service Provider",
            ["DSL", "Fiber optic", "No"],
            index=["DSL", "Fiber optic", "No"].index(default_internet)
        )
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(default_security)
        )

    with sc3:
        online_backup = st.selectbox(
            "Online Cloud Backup",
            ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(default_backup)
        )
        device_protection = st.selectbox(
            "Device Protection Plan",
            ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(default_protection)
        )

    with sc4:
        tech_support = st.selectbox(
            "Premium Tech Support",
            ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(default_support)
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(default_tv)
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(default_movies)
        )

    st.markdown("")
    predict_btn = st.button("🔮 Predict Customer Churn Risk", type="primary", use_container_width=True)

    if predict_btn:
        # Construct single-row DataFrame with exact schema
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
        stay_prob = probabilities[0] * 100
        churn_prob = probabilities[1] * 100

        st.markdown("### Prediction Result")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class='churn-alert-danger'>
                        <h3 style='color: #B91C1C; margin:0;'>⚠️ High Risk of Churn</h3>
                        <p style='margin: 8px 0 0 0; font-size: 1.1rem;'>
                            This customer is predicted to <strong>LEAVE (Churn)</strong> the service.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class='churn-alert-safe'>
                        <h3 style='color: #15803D; margin:0;'>✅ Low Risk / Loyal Customer</h3>
                        <p style='margin: 8px 0 0 0; font-size: 1.1rem;'>
                            This customer is predicted to <strong>STAY (Retained)</strong> with the company.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with res_col2:
            st.markdown(f"**Churn Probability:** `{churn_prob:.1f}%`")
            st.progress(float(churn_prob / 100))
            st.markdown(f"**Retention Probability:** `{stay_prob:.1f}%`")
            st.progress(float(stay_prob / 100))

        # Business recommendations
        st.markdown("#### 💡 Retention Insights & Recommended Actions")
        if prediction == 1:
            st.warning(
                "**Action Plan for Customer Retention:**\n"
                "- **Contract Incentive:** Transition the customer from a month-to-month contract to a 1-year or 2-year plan by offering a promotional discount.\n"
                "- **Value-Add Services:** Bundle complimentary Tech Support or Online Security.\n"
                "- **Payment Method Review:** Encourage automatic bank debit or credit card billing over electronic checks, which are correlated with higher churn."
            )
        else:
            st.success(
                "**Action Plan for Maintaining Loyalty:**\n"
                "- The customer demonstrates high loyalty indicators (longer tenure, stable contract, or bundled services).\n"
                "- Maintain periodic satisfaction check-ins and reward loyalty with anniversary perks."
            )

# =========================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
with tab2:
    st.markdown("### Exploratory Data Analysis (EDA) Visualizations")
    st.markdown("Key patterns discovered in the Telco dataset that inform machine learning predictions:")

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        st.markdown("#### 1. Overall Churn Distribution")
        p1 = os.path.join(FIGURES_DIR, "churn_distribution.png")
        if os.path.exists(p1):
            st.image(p1, caption="Figure 1: Churn vs Retained breakdown (26.5% churn rate)", use_container_width=True)
            st.info("💡 **Insight:** Approximately 26.5% of customers churned. This demonstrates class imbalance (~3:1 ratio), which is why we measure Precision, Recall, and F1-score rather than accuracy alone.")

    with eda_col2:
        st.markdown("#### 2. Customer Churn by Contract Type")
        p2 = os.path.join(FIGURES_DIR, "contract_churn.png")
        if os.path.exists(p2):
            st.image(p2, caption="Figure 2: Churn frequency across contract terms", use_container_width=True)
            st.info("💡 **Insight:** Month-to-month customers have by far the highest churn rate (~42%), whereas customers on One-Year (~11%) and Two-Year (~3%) contracts churn rarely.")

    eda_col3, eda_col4 = st.columns(2)

    with eda_col3:
        st.markdown("#### 3. Monthly Charges vs Churn")
        p3 = os.path.join(FIGURES_DIR, "monthly_charges_churn.png")
        if os.path.exists(p3):
            st.image(p3, caption="Figure 3: Density of Monthly Charges ($)", use_container_width=True)
            st.info("💡 **Insight:** Customers with higher monthly charges ($70 - $100) experience a significantly higher churn density compared to budget customers ($20 - $30).")

    with eda_col4:
        st.markdown("#### 4. Customer Tenure vs Churn")
        p4 = os.path.join(FIGURES_DIR, "tenure_churn.png")
        if os.path.exists(p4):
            st.image(p4, caption="Figure 4: Churn distribution across tenure in months", use_container_width=True)
            st.info("💡 **Insight:** Churn is heavily concentrated in the first 0 to 6 months. Customers who stay past month 24 develop high loyalty and rarely leave.")

# =========================================================
# TAB 3: MODEL PERFORMANCE & METRICS
# =========================================================
with tab3:
    st.markdown("### Model Evaluation Metrics (Test Set: 20% Unseen Data)")
    st.markdown("The Logistic Regression model was evaluated on 1,409 unseen customer test samples:")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{metadata['accuracy'] * 100:.2f}%</div>
                <div class='metric-label'>Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{metadata['precision'] * 100:.2f}%</div>
                <div class='metric-label'>Precision</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{metadata['recall'] * 100:.2f}%</div>
                <div class='metric-label'>Recall</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m4:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{metadata['f1_score'] * 100:.2f}%</div>
                <div class='metric-label'>F1-Score</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    eval_c1, eval_c2 = st.columns([1, 1])

    with eval_c1:
        st.markdown("#### 🔍 Confusion Matrix")
        p_cm = os.path.join(FIGURES_DIR, "confusion_matrix.png")
        if os.path.exists(p_cm):
            st.image(p_cm, caption="Confusion Matrix on Test Set", use_container_width=True)

    with eval_c2:
        st.markdown("#### 📖 Understanding the Evaluation Metrics (for BTech Viva / Exams)")
        st.markdown("""
        - **Accuracy (80.62%):** Percentage of total predictions that were correct:  
          $\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}$
        
        - **Precision (65.93%):** When the model predicts a customer will churn, how often is it right?  
          $\\text{Precision} = \\frac{TP}{TP + FP}$  
          *High precision prevents wasting retention budget on customers who wouldn't have left.*
        
        - **Recall (55.88%):** Out of all actual churners, what percentage did the model capture?  
          $\\text{Recall} = \\frac{TP}{TP + FN}$  
          *High recall ensures the company catches as many at-risk customers as possible.*
        
        - **F1-Score (60.49%):** Harmonic mean balancing Precision and Recall:  
          $\\text{F1} = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$
        """)

# =========================================================
# TAB 4: BTECH STUDENT GUIDE & DATASET EXPLORER
# =========================================================
with tab4:
    st.markdown("### 🎓 Complete Project Guide for BTech Data Science Students")
    st.markdown("""
    #### 1. Why Customer Churn Prediction?
    Acquiring a new customer costs **5x to 7x more** than retaining an existing one. Telecom companies use predictive churn models to identify customers at risk of cancellation before they actually leave, allowing targeted customer retention interventions.

    #### 2. The 6-Step Machine Learning Pipeline in this Project:
    1. **Data Loading:** Read `Telco-Customer-Churn.csv` containing 7,043 customer records.
    2. **Data Cleaning:**
       - Detect 11 whitespace strings in `TotalCharges` column.
       - Convert `TotalCharges` to `float64` and impute `0.0` for 0-tenure records.
       - Drop non-predictive `customerID`.
       - Map `Churn` to binary labels (`Yes` = 1, `No` = 0).
    3. **Exploratory Data Analysis (EDA):**
       - Uncover key factors driving churn: Contract duration, high monthly fees, short tenure.
    4. **Feature Engineering & Preprocessing Pipeline:**
       - `StandardScaler` for continuous numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`).
       - `OneHotEncoder(drop='first')` for categorical variables to prevent the dummy variable trap.
    5. **Model Training:**
       - Logistic Regression with L2 regularization and `max_iter=1000`.
       - Sigmoid activation function: $\\sigma(z) = \\frac{1}{1 + e^{-z}}$
    6. **Model Evaluation & Deployment:**
       - Evaluate using Stratified 80/20 train/test split.
       - Save trained pipeline with `joblib` and build a live web interface with `Streamlit`.
    """)

    st.markdown("---")
    st.markdown("#### 📂 Sample Dataset Explorer")
    if os.path.exists(DATA_PATH):
        raw_df = pd.read_csv(DATA_PATH)
        st.dataframe(raw_df.head(10), use_container_width=True)
        st.caption(f"Showing first 10 rows out of {len(raw_df):,} total customer records.")
