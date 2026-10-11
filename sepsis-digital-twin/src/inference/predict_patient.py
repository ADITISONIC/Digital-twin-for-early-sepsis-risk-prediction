import os
import pandas as pd
import numpy as np
import pickle
import logging
import gc
import shap
import matplotlib.pyplot as plt

from src.utils.helpers import load_config, set_seeds
from src.data.loader import load_patient_file
from src.features.build_features import build_dynamic_features_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_final_model_comparison(base_dir, config):
    """
    Combines the Logistic Regression, XGBoost, and LSTM holdout metric files
    into a single CSV table: results/metrics/final_model_comparison.csv
    """
    logger.info("Generating unified final model comparison table...")
    results_dir = os.path.join(base_dir, config["paths"]["results_dir"])
    metrics_dir = os.path.join(results_dir, "metrics")
    
    files = {
        "Logistic Regression": "logistic_regression_holdout_metrics.csv",
        "XGBoost": "xgboost_holdout_metrics.csv",
        "LSTM": "lstm_holdout_metrics.csv"
    }
    
    rows = []
    for model_name, filename in files.items():
        filepath = os.path.join(metrics_dir, filename)
        if not os.path.exists(filepath):
            logger.warning("Metric file %s not found. Skipping.", filename)
            continue
            
        df = pd.read_csv(filepath)
        row = df.iloc[0].to_dict()
        
        def fmt_ci(val, ci_str):
            if isinstance(val, (int, float)) and not pd.isna(val):
                out = f"{val:.4f}"
                if isinstance(ci_str, str) and len(ci_str) > 0:
                    out += f" {ci_str}"
                return out
            return "N/A"
            
        def fmt_warn(val, ci_str):
            if isinstance(val, (int, float)) and not pd.isna(val):
                out = f"{val:.1f}"
                if isinstance(ci_str, str) and len(ci_str) > 0:
                    out += f" {ci_str}"
                return out
            return "N/A"
            
        rows.append({
            "Model": model_name,
            "AUROC": fmt_ci(row.get("auroc"), row.get("auroc_95ci")),
            "AUPRC": fmt_ci(row.get("auprc"), row.get("auprc_95ci")),
            "Sensitivity": fmt_ci(row.get("sensitivity"), row.get("sensitivity_95ci")),
            "Specificity": fmt_ci(row.get("specificity"), row.get("specificity_95ci")),
            "Precision": f"{row.get('precision', np.nan):.4f}" if not pd.isna(row.get('precision')) else "N/A",
            "F1": f"{row.get('f1', np.nan):.4f}" if not pd.isna(row.get('f1')) else "N/A",
            "Brier": f"{row.get('brier', np.nan):.4f}" if not pd.isna(row.get('brier')) else "N/A",
            "Utility": fmt_ci(row.get("normalized_utility"), row.get("utility_95ci")),
            "Warning Time": fmt_warn(row.get("median_warning_time"), row.get("warning_time_95ci")),
            "False Alerts": f"{row.get('false_alerts_per_100h', np.nan):.2f}" if not pd.isna(row.get('false_alerts_per_100h')) else "N/A"
        })
        
    if len(rows) > 0:
        comparison_df = pd.DataFrame(rows)
        out_path = os.path.join(metrics_dir, "final_model_comparison.csv")
        comparison_df.to_csv(out_path, index=False)
        logger.info("Saved final model comparison to %s", out_path)
    else:
        logger.error("No metrics files were found to compile comparison table.")

def predict_patient(patient_id):
    config = load_config()
    set_seeds(config["random_seed"])
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    models_dir = os.path.join(base_dir, config["paths"]["models_dir"])
    results_dir = os.path.join(base_dir, config["paths"]["results_dir"])
    predictions_dir = os.path.join(results_dir, "predictions")
    figures_dir = os.path.join(results_dir, "figures")
    
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Load model artifact
    logger.info("Loading XGBoost model artifact for patient-level inference...")
    xgb_path = os.path.join(models_dir, "xgboost_model.pkl")
    with open(xgb_path, "rb") as f:
        artifact = pickle.load(f)
        
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    feature_cols = artifact["feature_cols"]
    threshold = artifact["best_threshold"]
    scaler = artifact.get("scaler", None)
    del artifact
    gc.collect()
    
    # 2. Load raw patient p100013 from Hospital B holdout
    patient_id="p100013"
    raw_file = os.path.join(base_dir, config["paths"]["raw_dir"], "training_setB", f"{patient_id}.psv")
    
    logger.info("Loading patient %s raw data...", patient_id)
    expected_cols = (
        config["variables"]["vitals"] + 
        config["variables"]["labs"] + 
        config["variables"]["demographics"] + 
        [config["data"]["target_col"]]
    )
    pat_df = load_patient_file(raw_file, patient_id, 'B', expected_cols)
    
    # 3. Preprocess causally and dynamically build features
    logger.info("Running preprocessor and feature engineering pipeline on patient data...")
    pat_transformed = preprocessor.transform([pat_df])
    pat_feat_dfs = build_dynamic_features_pipeline(
        pat_transformed,
        vitals=config["variables"]["vitals"],
        labs=config["variables"]["labs"]
    )
    processed_df = pat_feat_dfs[0]
    
    # 4. Scale features if applicable
    X = processed_df[feature_cols].values
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
        
    # 5. Risk prediction
    probs = model.predict_proba(X_scaled)[:, 1]
    processed_df["PredictedRisk"] = probs
    processed_df["SepsisAlert"] = (probs >= threshold).astype(int)
    
    # 6. Sepsis labels and warning times
    labels = processed_df[config["data"]["target_col"]].values
    hours = processed_df["ICULOS"].values
    
    has_sepsis = np.any(labels == 1)
    has_alert = np.any(processed_df["SepsisAlert"] == 1)
    
    first_alert_hour = None
    first_alert_idx = None
    if has_alert:
        first_alert_idx = np.where(processed_df["SepsisAlert"] == 1)[0][0]
        first_alert_hour = int(hours[first_alert_idx])
        
    actual_onset_hour = None
    warning_time = None
    if has_sepsis:
        t_label_onset = np.argmax(labels)
        # SepsisLabel is shifted 6 hours early.
        # The clinical onset hour is: t_onset = ArgMax(labels) + 6
        actual_onset_hour = int(hours[t_label_onset] + 6)
        
        if has_alert:
            warning_time = actual_onset_hour - first_alert_hour
            
    # 7. Compute SHAP values for explanation
    logger.info("Computing SHAP explanation values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(processed_df[feature_cols])
    
    # Get top 5 contributing features at first alert (or final hour if no alert)
    explanation_idx = first_alert_idx if first_alert_idx is not None else len(processed_df) - 1
    shap_for_hour = shap_values.values[explanation_idx]
    
    # Sort features by SHAP value descending (highest positive contributions)
    top_feature_indices = np.argsort(shap_for_hour)[::-1][:5]
    top_features = []
    for idx in top_feature_indices:
        feat_name = feature_cols[idx]
        feat_val = processed_df[feat_name].iloc[explanation_idx]
        shap_val = shap_for_hour[idx]
        top_features.append((feat_name, feat_val, shap_val))
        
    # Save CSV
    out_csv = os.path.join(predictions_dir, "example_patient_prediction.csv")
    output_df = processed_df[["PatientID", "ICULOS", "SepsisLabel", "PredictedRisk", "SepsisAlert"]]
    output_df.to_csv(out_csv, index=False)
    logger.info("Saved patient predictions to %s", out_csv)
    
    # Plot risk trajectory
    
    
    # Terminal Display
    print("\n" + "="*50)
    print("DIGITAL TWIN PATIENT DEMONSTRATION")
    print("="*50)
    print(f"Patient ID                    : {patient_id}")
    print(f"Number of ICU hours            : {len(processed_df)}")
    print(f"Final/current predicted risk   : {probs[-1]:.4f}")
    print(f"Alert threshold               : {threshold:.2f}")
    print(f"Alert occurred                : {has_alert}")
    if has_alert:
        print(f"First alert hour              : Hour {first_alert_hour}")
    if has_sepsis:
        print(f"Actual clinical onset hour    : Hour {actual_onset_hour}")
        if has_alert:
            print(f"Warning time                  : {warning_time} hours prior to clinical onset")
    print("\nTop 5 contributing features (SHAP):")
    for feat_name, val, shap_val in top_features:
        print(f" - {feat_name:25}: Value = {val:8.4f} (SHAP contribution = {shap_val:+8.4f})")
    print("="*50 + "\n")
    
    # Generate the comparison table
    generate_final_model_comparison(base_dir, config)
    return {
    "predictions": processed_df,
    "probs": probs,
    "threshold": threshold,
    "first_alert_hour": first_alert_hour,
    "actual_onset_hour": actual_onset_hour,
    "warning_time": warning_time,
    "top_features": top_features,
    "patient_id": patient_id
    }
def predict_realtime(patient_df):

    config = load_config()

    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )

    models_dir = os.path.join(
        base_dir,
        config["paths"]["models_dir"]
    )

    xgb_path = os.path.join(
        models_dir,
        "xgboost_model.pkl"
    )

    with open(xgb_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    feature_cols = artifact["feature_cols"]
    threshold = artifact["best_threshold"]

    scaler = artifact.get("scaler", None)

    pat_transformed = preprocessor.transform(
        [patient_df]
    )

    pat_feat_dfs = build_dynamic_features_pipeline(
        pat_transformed,
        vitals=config["variables"]["vitals"],
        labs=config["variables"]["labs"]
    )

    processed_df = pat_feat_dfs[0]

    X = processed_df[feature_cols].values

    if scaler is not None:
        X = scaler.transform(X)

    probs = model.predict_proba(X)[:, 1]

    processed_df["PredictedRisk"] = probs

    processed_df["SepsisAlert"] = (
        probs >= threshold
    ).astype(int)

    return {
        "predictions": processed_df,
        "risk": probs,
        "threshold": threshold
    }

def get_severity(risk):

    if risk < 0.30:
        return "LOW"

    elif risk < 0.60:
        return "MODERATE"

    elif risk < 0.80:
        return "HIGH"

    else:
        return "CRITICAL"


if __name__ == "__main__":
    predict_patient("p100013")
