from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.build_features import build_dynamic_features_pipeline  # noqa: E402

MODEL_PATH = ROOT / "models" / "xgboost_model.pkl"
RAW_B = ROOT / "data" / "raw" / "training_setB"

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

MODEL = artifact["model"]
PREPROCESSOR = artifact["preprocessor"]
FEATURE_COLS = artifact["feature_cols"]
THRESHOLD = float(artifact["best_threshold"])

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets"
]
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
MODEL_INPUTS = VITALS + LABS + DEMOGRAPHICS + ["SepsisLabel"]

app = FastAPI(title="Sepsis Digital Twin Demo API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    patient_id: str = "demo"
    hospital_id: str = "B"
    rows: List[Dict[str, Any]] = Field(min_length=1)

def _normalise_rows(rows: List[Dict[str, Any]], patient_id: str, hospital_id: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "ICULOS" not in df.columns:
        df["ICULOS"] = np.arange(1, len(df) + 1)
    for col in MODEL_INPUTS:
        if col not in df.columns:
            df[col] = np.nan
    df["PatientID"] = patient_id
    df["HospitalID"] = hospital_id
    for col in MODEL_INPUTS:
        if col != "SepsisLabel":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["SepsisLabel"] = pd.to_numeric(df["SepsisLabel"], errors="coerce").fillna(0).astype(int)
    return df.sort_values("ICULOS").reset_index(drop=True)

def _predict(df: pd.DataFrame) -> pd.DataFrame:
    transformed = PREPROCESSOR.transform([df])
    featured = build_dynamic_features_pipeline(
        transformed,
        vitals=VITALS,
        labs=LABS,
    )[0]
    X = featured[FEATURE_COLS].values
    probs = MODEL.predict_proba(X)[:, 1]
    featured["PredictedRisk"] = probs
    featured["SepsisAlert"] = (probs >= THRESHOLD).astype(int)
    return featured

def _risk(prob: float) -> str:
    if prob >= THRESHOLD:
        return "HIGH"
    if prob >= max(0.50, THRESHOLD * 0.6):
        return "WARNING"
    return "LOW"

def _latest_explanation(featured: pd.DataFrame) -> Dict[str, Any]:
    explainer = shap.TreeExplainer(MODEL)
    X = featured[FEATURE_COLS]
    vals = explainer.shap_values(X)
    if hasattr(vals, "values"):
        vals = vals.values
    vals = np.asarray(vals)
    i = len(featured) - 1
    row_vals = vals[i]
    top = np.argsort(np.abs(row_vals))[::-1][:10]
    items = []
    for j in top:
        items.append({
            "feature": FEATURE_COLS[j],
            "value": None if pd.isna(featured.iloc[i][FEATURE_COLS[j]]) else float(featured.iloc[i][FEATURE_COLS[j]]),
            "shap": float(row_vals[j]),
            "direction": "increases risk" if row_vals[j] > 0 else "decreases risk",
        })
    return {"base_value": float(explainer.expected_value), "features": items}

@app.get("/api/health")
def health():
    return {"status": "ok", "model": "XGBoost", "threshold": THRESHOLD}

@app.get("/api/config")
def config():
    return {
        "model": "XGBoost",
        "threshold": THRESHOLD,
        "inputs": MODEL_INPUTS,
        "vitals": VITALS,
        "labs": LABS,
        "demographics": DEMOGRAPHICS,
    }

@app.get("/api/patients")
def patients():
    if not RAW_B.exists():
        raise HTTPException(500, f"Hospital B data directory not found: {RAW_B}")
    out = []
    for p in sorted(RAW_B.glob("*.psv"))[:20000]:
        try:
            d = pd.read_csv(p, sep="|", nrows=1)
            out.append({
                "patient_id": p.stem,
                "age": None if pd.isna(d.iloc[0].get("Age")) else float(d.iloc[0]["Age"]),
                "gender": None if pd.isna(d.iloc[0].get("Gender")) else int(d.iloc[0]["Gender"]),
            })
        except Exception:
            continue
    return {"patients": out}

@app.get("/api/patients/{patient_id}")
def patient(patient_id: str):
    path = RAW_B / f"{patient_id}.psv"
    if not path.exists():
        raise HTTPException(404, "Patient not found in Hospital B")
    raw = pd.read_csv(path, sep="|")
    featured = _predict(_normalise_rows(raw.to_dict("records"), patient_id, "B"))
    return {
        "patient_id": patient_id,
        "hospital_id": "B",
        "rows": featured[VITALS + LABS + DEMOGRAPHICS + ["ICULOS", "SepsisLabel", "PredictedRisk", "SepsisAlert"]].replace({np.nan: None}).to_dict("records"),
        "threshold": THRESHOLD,
        "latest": {
            "probability": float(featured["PredictedRisk"].iloc[-1]),
            "risk": _risk(float(featured["PredictedRisk"].iloc[-1])),
            "alert": bool(featured["SepsisAlert"].iloc[-1]),
            "timestamp": int(featured["ICULOS"].iloc[-1]),
        },
        "explanation": _latest_explanation(featured),
    }

@app.post("/api/predict")
def predict(req: PredictionRequest):
    try:
        df = _normalise_rows(req.rows, req.patient_id, req.hospital_id)
        featured = _predict(df)
        p = float(featured["PredictedRisk"].iloc[-1])
        return {
            "patient_id": req.patient_id,
            "hospital_id": req.hospital_id,
            "timestamp": int(featured["ICULOS"].iloc[-1]),
            "probability": p,
            "risk": _risk(p),
            "alert": bool(featured["SepsisAlert"].iloc[-1]),
            "threshold": THRESHOLD,
            "explanation": _latest_explanation(featured),
        }
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}")
