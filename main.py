"""Customer Churn Prediction - FastAPI backend (PBADT304 Introduction to Data Science).

Run:   python main.py          (opens http://127.0.0.1:8000)
Data:  put any churn CSV next to this file (or in ./data) or upload it in the website.
       With no CSV, a synthetic Telco-style dataset is generated so everything still works.
"""
import glob
import io
import json
import threading
import webbrowser
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).parent
app = FastAPI(title="Customer Churn Prediction API")
S: dict = {}          # in-memory application state


def records(df: pd.DataFrame):
    """DataFrame -> JSON-safe list of dicts (NaN becomes null)."""
    return json.loads(df.to_json(orient="records"))


# ============================== 1-3. DATA REQUIREMENT & COLLECTION ============
def make_synthetic(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[.35, .45, .20])
    pay = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
    senior = rng.choice([0, 1], n, p=[.84, .16])
    support = rng.choice(["Yes", "No"], n, p=[.4, .6])
    gender = rng.choice(["Male", "Female"], n)
    monthly = (20 + 30 * (internet == "DSL") + 55 * (internet == "Fiber optic")
               + rng.normal(0, 6, n)).clip(18, 120).round(2)
    total = (monthly * tenure + rng.normal(0, 20, n)).clip(0).round(2)
    logit = (-0.5 - 0.035 * tenure + 0.02 * (monthly - 65)
             + 1.0 * (contract == "Month-to-month") - 0.8 * (contract == "Two year")
             + 0.5 * (internet == "Fiber optic") + 0.4 * (pay == "Electronic check")
             - 0.5 * (support == "Yes") + 0.3 * senior)
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)], "gender": gender,
        "SeniorCitizen": senior, "tenure": tenure, "Contract": contract,
        "InternetService": internet, "TechSupport": support, "PaymentMethod": pay,
        "MonthlyCharges": monthly, "TotalCharges": total.astype(str),
        "Churn": np.where(churn, "Yes", "No")})
    df.loc[rng.choice(n, 12, replace=False), "TotalCharges"] = " "      # dirty values
    return pd.concat([df, df.sample(20, random_state=1)], ignore_index=True)  # duplicates


def find_raw():
    for p in sorted(glob.glob(str(BASE / "*.csv")) + glob.glob(str(BASE / "data" / "*.csv"))):
        return pd.read_csv(p), f"File: {Path(p).name}"
    return make_synthetic(), "Synthetic demo data (no CSV found)"


# ============================== 4. DATA PREPARATION ===========================
def clean(df, target):
    log, d = [], df.copy()
    n0 = len(d)
    d = d.drop_duplicates()
    log.append(f"Removed {n0 - len(d)} duplicate rows")
    ids = [c for c in d.columns if c != target and not pd.api.types.is_numeric_dtype(d[c])
           and d[c].nunique() == len(d)]
    if ids:
        d = d.drop(columns=ids)
        log.append(f"Dropped identifier column(s): {', '.join(ids)}")
    for c in d.columns:
        if c != target and not pd.api.types.is_numeric_dtype(d[c]):
            conv = pd.to_numeric(d[c].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
            if conv.notna().mean() > 0.9:
                d[c] = conv
                log.append(f"Converted '{c}' from text to numeric")
    if not pd.api.types.is_numeric_dtype(d[target]):
        d[target] = d[target].astype(str).str.strip().str.lower().isin(
            ["yes", "1", "true", "churn", "y"]).astype(int)
        log.append(f"Encoded target '{target}' as Yes=1 / No=0")
    miss = int(d.drop(columns=[target]).isna().sum().sum())
    log.append(f"{miss} missing feature values are imputed (median / most frequent)")
    return d, log


# ============================== 5. EDA ========================================
def make_eda(d, target, num, cat):
    y = d[target]
    eda = {"counts": {"Stayed": int((y == 0).sum()), "Churned": int((y == 1).sum())},
           "numeric": {}, "categorical": {}}
    num = sorted(num, key=lambda c: d[c].nunique() <= 2)   # continuous columns first, 0/1 columns last
    for c in num:
        s = d[c].dropna()
        edges = np.histogram_bin_edges(s, bins=12)
        lab = [f"{(edges[i] + edges[i + 1]) / 2:.1f}" for i in range(len(edges) - 1)]
        st_, ch_ = (np.histogram(d.loc[y == k, c].dropna(), bins=edges)[0].tolist() for k in (0, 1))
        box = {}
        for k, name in ((0, "Stayed"), (1, "Churned")):
            g = d.loc[y == k, c].dropna()
            box[name] = [round(float(v), 2) for v in
                         (g.min(), g.quantile(.25), g.median(), g.quantile(.75), g.max())] if len(g) else []
        eda["numeric"][c] = {"labels": lab, "stayed": st_, "churned": ch_, "box": box}
    for c in cat:
        if d[c].nunique() <= 30:
            g = d.groupby(c)[target].agg(["mean", "size"]).sort_values("size", ascending=False).head(12)
            eda["categorical"][c] = [{"category": str(i), "rate": round(float(r["mean"]), 4),
                                      "n": int(r["size"])} for i, r in g.iterrows()]
    cols = num + [target]
    corr = d[cols].corr().fillna(0)
    eda["corr"] = {"cols": cols, "matrix": corr.round(2).values.tolist()}
    sc = d[cols].dropna()
    eda["scatter"] = {"cols": num, "rows": records(sc.sample(min(500, len(sc)), random_state=1)
                                                    .rename(columns={target: "_churn"}))}
    if num:
        desc = d[num].describe().T
        desc["skew"] = d[num].skew()
        eda["describe"] = records(desc.round(3).reset_index().rename(columns={"index": "feature"}))
    rows = []
    for c in num:
        t, p = stats.ttest_ind(d.loc[y == 1, c].dropna(), d.loc[y == 0, c].dropna(), equal_var=False)
        rows.append({"feature": c, "test": "Welch t-test", "statistic": float(t), "p_value": float(p)})
    for c in cat:
        if d[c].nunique() > 1:
            chi, p, _, _ = stats.chi2_contingency(pd.crosstab(d[c], y))
            rows.append({"feature": c, "test": "Chi-square", "statistic": float(chi), "p_value": float(p)})
    rows.sort(key=lambda r: (np.isnan(r["p_value"]), r["p_value"]))
    for r in rows:
        r["significant"] = "Yes" if r["p_value"] < 0.05 else "No"
    eda["tests"] = rows
    return eda


# ============================== 6-7. MODELLING & EVALUATION ===================
def train(d, target, test_size, seed):
    X, y = d.drop(columns=[target]), d[target]
    nums = list(X.select_dtypes("number").columns)
    cats = [c for c in X.columns if c not in nums]
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), nums),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), cats)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    algos = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8,
                                                class_weight="balanced", random_state=seed)}
    models, rows, extra = {}, [], {}
    for name, m in algos.items():
        pipe = Pipeline([("pre", pre), ("model", m)]).fit(Xtr, ytr)
        pred, prob = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, prob)
        idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 60)).astype(int)
        rows.append({"model": name, "accuracy": accuracy_score(yte, pred),
                     "precision": precision_score(yte, pred, zero_division=0),
                     "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
                     "roc_auc": roc_auc_score(yte, prob),
                     "rmse": float(np.sqrt(np.mean((prob - yte.values) ** 2)))})
        extra[name] = {"confusion": confusion_matrix(yte, pred).tolist(),
                       "roc": [{"x": float(fpr[i]), "y": float(tpr[i])} for i in idx]}
        models[name] = {"pipe": pipe, "pred": pred}
    rf = models["Random Forest"]["pipe"]
    names = [n.split("__", 1)[-1] for n in rf["pre"].get_feature_names_out()]
    imp = pd.Series(rf["model"].feature_importances_, index=names).nlargest(12)
    best = max(rows, key=lambda r: r["f1"])["model"]
    payload = {"metrics": rows, "details": extra, "best": best,
               "importance": [{"feature": k, "value": round(float(v), 4)} for k, v in imp.items()],
               "train_rows": len(Xtr), "test_rows": len(Xte)}
    return models, payload, yte


# ============================== BUILD / STATE =================================
def build(raw, source, target=None, test_size=0.2, seed=42):
    if target not in raw.columns:
        guess = [c for c in raw.columns if "churn" in c.lower()]
        target = guess[0] if guess else raw.columns[-1]
    d, log = clean(raw, target)
    num = [c for c in d.select_dtypes("number").columns if c != target]
    cat = [c for c in d.columns if c not in num + [target]]
    models, payload, yte = train(d, target, test_size, seed)
    feats = []
    for c in [x for x in d.columns if x != target]:
        s = d[c].dropna()
        if c in num and s.nunique() > 2:
            feats.append({"name": c, "kind": "number", "min": float(s.min()), "max": float(s.max()),
                          "median": float(s.median())})
        else:
            opts = sorted(s.unique().tolist()) if c in num else sorted(s.astype(str).unique().tolist())
            feats.append({"name": c, "kind": "choice", "options": opts})
    S.update(raw=raw, data=d, log_steps=log, source=source, target=target, num=num, cat=cat,
             models=models, results=payload, y_test=yte, test_size=test_size, seed=seed,
             schema=feats, eda=make_eda(d, target, num, cat), pred_log=[])


def init():
    raw, src = find_raw()
    build(raw, src)


class PredictIn(BaseModel):
    algo: str
    inputs: dict


class FeedbackIn(BaseModel):
    id: int
    actual: Optional[str] = None


class RetrainIn(BaseModel):
    test_size: float = 0.2
    seed: int = 42


# ============================== API ===========================================
@app.get("/api/overview")
def overview():
    raw, d, t = S["raw"], S["data"], S["target"]
    cols = [{"column": c, "role": "TARGET" if c == t else "Feature", "dtype": str(raw[c].dtype),
             "unique": int(raw[c].nunique()), "missing": int(raw[c].isna().sum()),
             "example": str(raw[c].iloc[0])} for c in raw.columns]
    return {"source": S["source"], "target": t, "rows": len(raw), "cols": raw.shape[1],
            "missing": int(raw.isna().sum().sum()), "duplicates": int(raw.duplicated().sum()),
            "churn_rate": float(d[t].mean()), "columns": cols, "head": records(raw.head(15))}


@app.get("/api/preparation")
def preparation():
    raw, d = S["raw"], S["data"]
    return {"log": S["log_steps"], "before": [len(raw), raw.shape[1]], "after": [len(d), d.shape[1]],
            "missing": [{"column": c, "missing": int(v)} for c, v in d.isna().sum().items()]}


@app.get("/api/cleaned.csv")
def cleaned_csv():
    return Response(S["data"].to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=cleaned_churn.csv"})


@app.get("/api/eda")
def eda():
    return S["eda"]


@app.get("/api/models")
def models():
    return {**S["results"], "test_size": S["test_size"], "seed": S["seed"]}


@app.get("/api/schema")
def schema():
    return {"features": S["schema"], "algos": list(S["models"]), "best": S["results"]["best"]}


@app.post("/api/predict")
def predict(body: PredictIn):
    if body.algo not in S["models"]:
        raise HTTPException(400, "Unknown model")
    feats = [f["name"] for f in S["schema"]]
    row = {f: body.inputs.get(f) for f in feats}
    df = pd.DataFrame([row])
    for c in S["num"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    p = float(S["models"][body.algo]["pipe"].predict_proba(df)[0, 1])
    risk = "High" if p >= .6 else "Medium" if p >= .35 else "Low"
    action = {"High": "Offer a retention discount or contract upgrade now.",
              "Medium": "Send a satisfaction survey and keep monitoring.",
              "Low": "No action needed - customer looks stable."}[risk]
    entry = {"id": len(S["pred_log"]) + 1, "algo": body.algo, "probability": round(p, 3),
             "predicted": "Churned" if p >= .5 else "Stayed", "actual": None}
    S["pred_log"].append(entry)
    return {"probability": p, "risk": risk, "action": action, "id": entry["id"]}


@app.get("/api/monitoring")
def monitoring():
    best = S["results"]["best"]
    pred, true = S["models"][best]["pred"], S["y_test"].values
    size = max(30, len(true) // 8)
    acc = [float(accuracy_score(true[i:i + size], pred[i:i + size]))
           for i in range(0, len(true) - size + 1, size)]
    return {"model": best, "batch_size": size, "accuracy": acc}


@app.get("/api/log")
def get_log():
    done = [e for e in S["pred_log"] if e["actual"]]
    live = (sum(e["predicted"] == e["actual"] for e in done) / len(done)) if done else None
    return {"rows": S["pred_log"], "live_accuracy": live, "n_feedback": len(done)}


@app.post("/api/feedback")
def feedback(body: FeedbackIn):
    for e in S["pred_log"]:
        if e["id"] == body.id:
            e["actual"] = body.actual if body.actual in ("Churned", "Stayed") else None
            return {"ok": True}
    raise HTTPException(404, "Prediction not found")


@app.post("/api/retrain")
def retrain(body: RetrainIn):
    ts = min(max(body.test_size, 0.1), 0.4)
    build(S["raw"], S["source"], S["target"], ts, body.seed)
    return {"ok": True, "best": S["results"]["best"]}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        raw = pd.read_csv(io.BytesIO(await file.read()))
        build(raw, f"Uploaded: {file.filename}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not use this CSV: {e}")
    return {"ok": True, "target": S["target"]}


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
init()

if __name__ == "__main__":
    import uvicorn
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
