import os
import pickle
import logging
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

from src.utils.helpers import load_config
from src.features.build_features import build_dynamic_features_pipeline


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

config = load_config()

MODEL_PATH = (
    BASE_DIR
    / config["paths"]["models_dir"]
    / "xgboost_model.pkl"
)

RISK_GRAPH_PATH = (
    BASE_DIR
    / config["paths"]["results_dir"]
    / "figures"
    / "example_patient_risk_trajectory.png"
)

FIGURES_DIR = (

    Path(__file__).resolve().parent

    / "results"

    / "figures"

)


# --------------------------------------------------
# FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="Sepsis Digital Twin API",
    description="Real-time sepsis prediction with SHAP explainability",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class PredictionRequest(BaseModel):
    patient_data: list[dict[str, Any]]


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model_artifact():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"XGBoost model artifact not found: {MODEL_PATH}"
        )

    with open(MODEL_PATH, "rb") as file:
        artifact = pickle.load(file)

    required_keys = {
        "model",
        "preprocessor",
        "feature_cols",
        "best_threshold",
    }

    missing = required_keys - set(artifact.keys())

    if missing:
        raise ValueError(
            f"Model artifact is missing keys: {sorted(missing)}"
        )

    return artifact


# Load once when the API module starts.
try:
    ARTIFACT = load_model_artifact()

    MODEL = ARTIFACT["model"]
    PREPROCESSOR = ARTIFACT["preprocessor"]
    FEATURE_COLS = ARTIFACT["feature_cols"]
    THRESHOLD = float(ARTIFACT["best_threshold"])
    SCALER = ARTIFACT.get("scaler")

    EXPLAINER = shap.TreeExplainer(MODEL)

    logger.info("XGBoost model and SHAP explainer loaded successfully.")

except Exception:
    logger.exception("Unable to initialize the model.")
    raise


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_severity(risk: float) -> str:
    if risk < 0.30:
        return "LOW"
    elif risk < 0.60:
        return "MODERATE"
    elif risk < 0.80:
        return "HIGH"
    return "CRITICAL"


def clean_patient_data(patient_df: pd.DataFrame) -> pd.DataFrame:
    patient_df = patient_df.copy()
    patient_df.columns = patient_df.columns.str.strip()

    patient_df = patient_df.replace(
        ["", " ", "NaN", "nan", "NAN", "None", "null", "NULL"],
        np.nan,
    )

    identifier_columns = {"PatientID", "patient_id"}

    for col in patient_df.columns:
        if col not in identifier_columns:
            patient_df[col] = pd.to_numeric(
                patient_df[col],
                errors="coerce",
            )

    patient_df = patient_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return patient_df


def json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if value is None or value is pd.NA:
        return None

    if pd.isna(value):
        return None

    return value


@app.get("/images/shap-importance")

def get_shap_importance():

    image_path = FIGURES_DIR / "shap_importance.png"

    if not image_path.exists():

        raise HTTPException(status_code=404, detail="SHAP importance image not found")

    return FileResponse(str(image_path), media_type="image/png")

@app.get("/images/shap-summary")

def get_shap_summary():

    image_path = FIGURES_DIR / "shap_summary.png"

    if not image_path.exists():

        raise HTTPException(status_code=404, detail="SHAP summary image not found")

    return FileResponse(str(image_path), media_type="image/png")


# --------------------------------------------------
# PREDICTION AND FEATURE ENGINEERING
# --------------------------------------------------

def prepare_patient_features(patient_df: pd.DataFrame):
    """
    Run the same preprocessing and feature engineering pipeline
    used by the project's existing real-time inference function.
    """

    transformed = PREPROCESSOR.transform([patient_df])

    feature_dfs = build_dynamic_features_pipeline(
        transformed,
        vitals=config["variables"]["vitals"],
        labs=config["variables"]["labs"],
    )

    if not feature_dfs:
        raise ValueError("Feature engineering returned no patient data.")

    processed_df = feature_dfs[0].copy()

    missing_features = [
        col for col in FEATURE_COLS
        if col not in processed_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Generated features are missing required model columns: "
            f"{missing_features}"
        )

    X = processed_df[FEATURE_COLS].copy()

    # Keep feature names for SHAP, while applying the trained scaler
    # to the values used by the model.
    if SCALER is not None:
        X_model = SCALER.transform(X)
        X_model = pd.DataFrame(
            X_model,
            columns=FEATURE_COLS,
            index=X.index,
        )
    else:
        X_model = X.copy()

    return processed_df, X_model


# --------------------------------------------------
# SHAP EXPLANATION
# --------------------------------------------------

def explain_latest_observation(X_model: pd.DataFrame) -> dict:
    """
    Explain the latest observation using the trained model.

    Positive SHAP values push the model output toward the positive
    class; negative values push it away from the positive class.

    For binary XGBoost models, the explained output may be raw margin
    rather than probability. SHAP values are contributions to the
    explainer's output, not direct percentage-point changes in risk.
    """

    if X_model.empty:
        return {
            "available": False,
            "message": "No feature rows available for explanation.",
            "features": [],
        }

    latest_row = X_model.iloc[[-1]]

    shap_result = EXPLAINER(latest_row)

    values = np.asarray(shap_result.values)

    # Support common binary-classification SHAP output shapes.
    if values.ndim == 3:
        # Typical shape: (samples, features, classes)
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        else:
            values = values[:, :, -1]

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    shap_values = values[0]
    feature_values = latest_row.iloc[0].to_numpy()

    if len(shap_values) != len(FEATURE_COLS):
        raise ValueError(
            "SHAP output length does not match the model feature list."
        )

    contributions = []

    for index, feature_name in enumerate(FEATURE_COLS):
        contributions.append({
            "feature": str(feature_name),
            "value": float(feature_values[index]),
            "shap_value": float(shap_values[index]),
        })

    # Strongest positive contributors first.
    positive = sorted(
        [item for item in contributions if item["shap_value"] > 0],
        key=lambda item: item["shap_value"],
        reverse=True,
    )[:5]

    # Strongest negative contributors first.
    negative = sorted(
        [item for item in contributions if item["shap_value"] < 0],
        key=lambda item: item["shap_value"],
    )[:5]

    return {
        "available": True,
        "explained_output": "model_output",
        "positive_contributors": positive,
        "negative_contributors": negative,
        "top_features": sorted(
            contributions,
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )[:5],
    }


# --------------------------------------------------
# HEALTH CHECKS
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Sepsis Digital Twin API is running",
        "status": "healthy",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "shap_enabled": EXPLAINER is not None,
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        patient_df = pd.DataFrame(request.patient_data)

        if patient_df.empty:
            raise HTTPException(
                status_code=400,
                detail="No patient observations were provided.",
            )

        patient_df = clean_patient_data(patient_df)

        # The target must not be used as a model input.
        target_col = config["data"]["target_col"]
        patient_features_df = patient_df.drop(
            columns=[target_col],
            errors="ignore",
        )

        processed_df, X_model = prepare_patient_features(
            patient_features_df
        )

        probabilities = MODEL.predict_proba(X_model)[:, 1]

        if len(probabilities) == 0:
            raise ValueError("The model returned no predictions.")

        latest_risk = float(probabilities[-1])

        if not np.isfinite(latest_risk):
            raise ValueError("The model returned an invalid risk value.")

        latest_risk = float(np.clip(latest_risk, 0.0, 1.0))

        latest_hour = None

        if "ICULOS" in processed_df.columns:
            hour_value = processed_df["ICULOS"].iloc[-1]

            if pd.notna(hour_value):
                latest_hour = int(hour_value)

        # Explain the latest hourly prediction.
        shap_explanation = explain_latest_observation(X_model)

        response = {
            "risk": latest_risk,
            "risk_percentage": round(latest_risk * 100, 2),
            "threshold": THRESHOLD,
            "alert": bool(latest_risk >= THRESHOLD),
            "severity": get_severity(latest_risk),
            "hour": latest_hour,
            "shap": shap_explanation,
        }

        logger.info(
            "Prediction complete: risk=%.4f, severity=%s",
            latest_risk,
            response["severity"],
        )

        return json_safe(response)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# --------------------------------------------------
# RISK TRAJECTORY GRAPH
# --------------------------------------------------

@app.get("/risk-trajectory")
def get_risk_trajectory():
    if not RISK_GRAPH_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Risk trajectory graph not found: {RISK_GRAPH_PATH}",
        )

    return FileResponse(
        path=str(RISK_GRAPH_PATH),
        media_type="image/png",
        filename=RISK_GRAPH_PATH.name,
    )

from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import FileResponse

# Resolve paths relative to backend.py
BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "results" / "figures"


@app.get("/images/shap-importance")
def get_shap_importance():
    image_path = FIGURES_DIR / "shap_importance.png"

    if not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Image not found: {image_path}"
        )

    return FileResponse(
        path=str(image_path),
        media_type="image/png"
    )


@app.get("/images/shap-summary")
def get_shap_summary():
    image_path = FIGURES_DIR / "shap_summary.png"

    if not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Image not found: {image_path}"
        )

    return FileResponse(
        path=str(image_path),
        media_type="image/png"
    )


@app.get("/risk-trajectory")
def get_risk_trajectory():
    image_path = FIGURES_DIR / "example_patient_risk_trajectory.png"

    if not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Risk trajectory image not found"
        )

    return FileResponse(
        path=str(image_path),
        media_type="image/png"
    )


@app.head("/risk-trajectory")
def check_risk_trajectory():
    image_path = FIGURES_DIR / "example_patient_risk_trajectory.png"

    if not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Risk trajectory image not found"
        )

    return None
# --------------------------------------------------
# RUN SERVER DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )