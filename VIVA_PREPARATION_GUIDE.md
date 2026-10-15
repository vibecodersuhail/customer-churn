# 🎓 B.Tech Data Science Viva Voce & Interview Preparation Guide
**Project Title:** Customer Churn Prediction Using Data Science and Machine Learning  
**Target Audience:** Undergraduate Data Science / CSE / AI-ML Students  

---

## 📌 Part 1: Mathematical Foundations of the Model

### 1. Why Can’t We Use Linear Regression for Classification?
- **Range Mismatch:** Linear regression outputs unbounded values: $\hat{y} \in (-\infty, +\infty)$. For probability, we require outputs strictly bounded: $P(Y=1|X) \in [0, 1]$.
- **Sensitivity to Outliers:** A distant outlier can dramatically shift the linear regression hyperplane, severely altering the decision boundary.
- **Heteroscedasticity:** Binary target residuals violate the constant variance assumption of Ordinary Least Squares (OLS).

---

### 2. The Sigmoid (Logistic) Function
Logistic Regression transforms a linear combination of input features into a valid probability using the **Sigmoid function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where:
$$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$

- When $z \to +\infty \implies \sigma(z) \to 1$
- When $z \to -\infty \implies \sigma(z) \to 0$
- When $z = 0 \implies \sigma(0) = 0.5$ (Default Decision Threshold)

---

### 3. Odds and the Logit Transformation
The **Odds** of an event is the ratio of the probability that the event occurs to the probability that it does not:

$$\text{Odds} = \frac{P}{1 - P}$$

Taking the natural logarithm of Odds gives the **Logit**:

$$\ln\left(\frac{P}{1 - P}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$$

> **Key Examiner Takeaway:** Logistic regression is linear in the **log-odds** space, even though it is non-linear in the probability space!

---

### 4. Loss Function: Binary Cross-Entropy (Log-Loss)
We do **not** use Mean Squared Error (MSE) because it creates a non-convex surface with many local minima. Instead, we use **Maximum Likelihood Estimation (MLE)**, minimizing the **Binary Cross-Entropy Loss**:

$$\mathcal{L}(\beta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

- If actual $y = 1$ and predicted $\hat{p} \to 0$, Loss $\to \infty$ (heavy penalty).
- If actual $y = 0$ and predicted $\hat{p} \to 1$, Loss $\to \infty$ (heavy penalty).

---

## 📌 Part 2: Feature Engineering & Preprocessing

### 1. What is the "Dummy Variable Trap"?
When one-hot encoding a categorical feature with $k$ unique categories, creating $k$ binary columns introduces **perfect multicollinearity** because:
$$\sum_{j=1}^{k} D_j = 1$$
This causes the feature matrix $(X^T X)$ to be singular (non-invertible).  
**Solution:** Set `OneHotEncoder(drop='first')`, which creates $k - 1$ columns. The dropped column becomes the reference baseline.

### 2. Why is `StandardScaler` Essential for Logistic Regression?
Standardizes features to zero mean ($\mu = 0$) and unit standard deviation ($\sigma = 1$):
$$z = \frac{x - \mu}{\sigma}$$
- Ensures gradient descent steps converge smoothly and symmetrically.
- Prevents features with huge raw values (e.g., `TotalCharges` $\approx \$8,000$) from dominating features with small scales (e.g., `tenure` $\in [0, 72]$).
- L2 Regularization assumes all feature weights are on comparable scales.

### 3. What is Data Leakage and How Did We Prevent It?
- **Data Leakage** occurs when information from outside the training dataset (such as mean and variance of the test set) is used to fit transformers or create models.
- **Our Prevention:** We wrapped `ColumnTransformer` inside an `sklearn.pipeline.Pipeline`. The transformers are fitted **strictly on `X_train`** during `.fit()` and merely transformed on `X_test` during `.predict()`.

---

## 📌 Part 3: Model Evaluation & Business Tradeoffs

| Metric | Formula | Test Result | Business Significance |
| :--- | :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | **80.62%** | Overall correctness across both classes. |
| **Precision** | $\frac{TP}{TP + FP}$ | **65.93%** | Minimizes wasted marketing spend on customers who were never going to leave anyway. |
| **Recall** | $\frac{TP}{TP + FN}$ | **55.88%** | Maximizes detection of at-risk customers so they can be retained in time. |
| **F1-Score** | $2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$ | **60.49%** | Harmonic balance between Precision and Recall under class imbalance. |

---

## 📌 Part 4: Top 15 Viva Voce Questions & Model Answers

#### Q1: What dataset did you use, and what is its target variable?
> **Answer:** We used the IBM Telco Customer Churn dataset with 7,043 rows and 21 columns. The target variable is `Churn`, indicating whether a customer canceled their telecom subscription (`Yes` = 1, `No` = 0).

#### Q2: What data anomalies did you encounter and how did you resolve them?
> **Answer:** The `TotalCharges` column contained 11 blank space strings (`" "`) for new customers with `tenure = 0`. We parsed them using `pd.to_numeric(errors='coerce')` and imputed `0.0`, reflecting zero accumulated charges. We also dropped `customerID` as it has zero predictive value.

#### Q3: Why is class imbalance important in churn prediction?
> **Answer:** In our dataset, ~73.5% of customers stayed and ~26.5% churned. A naive baseline model predicting "Stay" for everyone would score 73.5% accuracy while catching 0% of churners. This is why we rely on Precision, Recall, and F1-Score rather than accuracy alone.

#### Q4: Why did you stratify the train-test split?
> **Answer:** Setting `stratify=y` guarantees that the 80% training set and 20% test set both preserve the exact 73.5% : 26.5% ratio of retained to churned customers, preventing sampling bias.

#### Q5: What are the strongest indicators of customer churn according to your model?
> **Answer:** The top positive drivers of churn are having a **Month-to-month contract**, **Fiber optic internet service**, and paying by **Electronic check**. The strongest protective factors against churn are **Two-Year contracts** and **high tenure**.

#### Q6: How does the Streamlit web application work?
> **Answer:** Streamlit provides a user-friendly reactive web interface. It accepts customer inputs from form dropdowns and sliders, packages them into a single-row Pandas DataFrame, and passes them to our serialized `Pipeline.predict()` and `Pipeline.predict_proba()` to compute live churn probabilities and recommended retention strategies.

#### Q7: Why did you use `joblib` instead of `pickle`?
> **Answer:** `joblib` is optimized for NumPy arrays and Scikit-Learn estimators containing large numerical weight matrices, providing faster serialization and lower memory consumption than standard Python `pickle`.

---

## 📌 Part 5: Project Presentation Checklist
- [x] Dataset downloaded and verified (`data/Telco-Customer-Churn.csv`)
- [x] Full training script operational (`eda_and_training.py`)
- [x] Academic Jupyter notebook ready (`Customer_Churn_Analysis.ipynb`)
- [x] High-resolution EDA figures saved (`figures/`)
- [x] Interactive web interface running (`app.py` / `run_app.bat`)
- [x] Automated test suite passing (`test_project.py` - 4/4 tests passed)
