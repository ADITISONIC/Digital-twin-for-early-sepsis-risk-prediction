import os
import pandas as pd
import numpy as np
import pickle
import logging
import gc
import json
from xgboost import XGBClassifier

from src.utils.helpers import load_config, set_seeds
from src.data.loader import load_cohort_data
from src.preprocessing.pipeline import CausalPreprocessor
from src.features.build_features import build_dynamic_features_pipeline
from src.evaluation.metrics import (
    calculate_standard_metrics,
    compute_cohort_utility,
    calculate_calibration_stats,
    calculate_warning_time,
    calculate_alarm_burden
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_xgboost():
    config = load_config()
    set_seeds(config["random_seed"])
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    splits_dir = os.path.join(base_dir, config["paths"]["splits_dir"])
    models_dir = os.path.join(base_dir, config["paths"]["models_dir"])
    results_dir = os.path.join(base_dir, config["paths"]["results_dir"])
    results_tables_dir = os.path.join(results_dir, "tables")
    results_metrics_dir = os.path.join(results_dir, "metrics")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_tables_dir, exist_ok=True)
    os.makedirs(results_metrics_dir, exist_ok=True)
    
    # 1. Load patient ID lists
    train_ids = pd.read_csv(os.path.join(splits_dir, "train_patients.csv"))["PatientID"].tolist()
    val_ids = pd.read_csv(os.path.join(splits_dir, "validation_patients.csv"))["PatientID"].tolist()
    
    logger.info("Loaded patient IDs: Train=%d, Val=%d", len(train_ids), len(val_ids))
    
    # 2. Load raw cohort data (list of DataFrames)
    train_dfs = load_cohort_data(train_ids, 'A', config)
    
    # 3. Fit and apply CausalPreprocessor on Train
    preprocessor = CausalPreprocessor(
        vitals=config["variables"]["vitals"],
        labs=config["variables"]["labs"],
        demographics=config["variables"]["demographics"],
        max_forward_fill_gap=config["preprocessing"]["max_forward_fill_gap"]
    )
    preprocessor.fit(train_dfs)
    
    train_dfs = preprocessor.transform(train_dfs)
    gc.collect()
    
    # 4. Build Dynamic Features on Train
    train_dfs = build_dynamic_features_pipeline(
        train_dfs,
        vitals=config["variables"]["vitals"],
        labs=config["variables"]["labs"]
    )
    gc.collect()
    
    # Define feature columns
    exclude_cols = ["PatientID", "HospitalID", "SepsisLabel", "Unit1", "Unit2"]
    sample_df = train_dfs[0]
    feature_cols = [c for c in sample_df.columns if c not in exclude_cols]
    
    # Extract training matrices and free memory of train list
    X_train = np.vstack([df[feature_cols].values for df in train_dfs])
    y_train = np.concatenate([df[config["data"]["target_col"]].values for df in train_dfs])
    negatives = (y_train == 0).sum()
    positives = (y_train == 1).sum()
    scale_pos_weight = negatives / positives
    xgb_params = config["xgb_params"].copy()
    xgb_params["scale_pos_weight"] = scale_pos_weight
    del train_dfs
    gc.collect()
    
    # 5. Load and process validation cohort data
    val_dfs = load_cohort_data(val_ids, 'A', config)
    val_dfs = preprocessor.transform(val_dfs)
    gc.collect()
    
    val_dfs = build_dynamic_features_pipeline(
        val_dfs,
        vitals=config["variables"]["vitals"],
        labs=config["variables"]["labs"]
    )
    gc.collect()
    
    # Reconstruct patient-level predictions and labels for validation before deleting val list
    val_patient_labels = [df[config["data"]["target_col"]].values for df in val_dfs]
    lengths = [len(df) for df in val_dfs]
    indices = np.cumsum([0] + lengths)
    
    X_val = np.vstack([df[feature_cols].values for df in val_dfs])
    y_val = np.concatenate([df[config["data"]["target_col"]].values for df in val_dfs])
    
    del val_dfs
    gc.collect()
    
    # 6. Class imbalance adjustment
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info("Class imbalance ratio (Neg/Pos hours): %.2f", imbalance_ratio)
    
    # 7. Grid search over key hyperparameters (max_depth and learning_rate)
    logger.info("Starting sensible hyperparameter tuning...")
    hyperparameter_grid = [
        {"max_depth": 4, "learning_rate": 0.05},
        {"max_depth": 4, "learning_rate": 0.1},
        {"max_depth": 6, "learning_rate": 0.05},
        {"max_depth": 6, "learning_rate": 0.1}
    ]
    
    best_val_auc = 0.0
    best_params = None
    best_xgb_model = None
    
    for params in hyperparameter_grid:
        logger.info("Tuning params: %s", params)
        model = XGBClassifier(
            n_estimators=100,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            scale_pos_weight=imbalance_ratio,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config["random_seed"],
            eval_metric="logloss",
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate on validation using AUROC
        probs = model.predict_proba(X_val)[:, 1]
        try:
            from sklearn.metrics import roc_auc_score
            auc_score = roc_auc_score(y_val, probs)
        except:
            auc_score = 0.5
            
        logger.info("  Validation AUROC: %.4f", auc_score)
        if auc_score > best_val_auc:
            best_val_auc = auc_score
            best_params = params
            best_xgb_model = model
            
    logger.info("Best parameters: %s, Best Validation AUROC: %.4f", best_params, best_val_auc)
    
    # Free X_train to save memory
    del X_train
    gc.collect()
    
    # 8. Threshold optimization on validation utility
    val_probs = best_xgb_model.predict_proba(X_val)[:, 1]
    
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_ut = -999.0
    
    logger.info("Optimizing threshold on validation utility...")
    for th in thresholds:
        val_preds_flat = (val_probs >= th).astype(int)
        cohort_preds = [val_preds_flat[indices[i]:indices[i+1]] for i in range(len(lengths))]
        
        ut, _ = compute_cohort_utility(val_patient_labels, cohort_preds)
        if ut > best_ut:
            best_ut = ut
            best_thresh = th
            
    logger.info("Best validation utility: %.4f at threshold %.2f", best_ut, best_thresh)
    
    # Save threshold selection
    threshold_data = {
        "best_threshold": float(best_thresh),
        "validation_utility": float(best_ut),
        "best_parameters": best_params
    }
    with open(os.path.join(results_dir, "threshold_selection.json"), "w") as f:
        json.dump(threshold_data, f, indent=4)
    logger.info("Saved threshold_selection.json")
    
    # Final evaluation metrics at best threshold
    val_preds_flat = (val_probs >= best_thresh).astype(int)
    cohort_preds = [val_preds_flat[indices[i]:indices[i+1]] for i in range(len(lengths))]
    cohort_probs = [val_probs[indices[i]:indices[i+1]] for i in range(len(lengths))]
    
    std_metrics = calculate_standard_metrics(y_val, val_probs, val_preds_flat)
    slope, intercept = calculate_calibration_stats(y_val, val_probs)
    warning_times = calculate_warning_time(val_patient_labels, cohort_preds)
    false_alert_rate, overall_alert_rate = calculate_alarm_burden(val_patient_labels, cohort_preds)
    
    metrics_summary = {
        "model": "xgboost",
        "threshold": best_thresh,
        "auroc": std_metrics["auroc"],
        "auprc": std_metrics["auprc"],
        "sensitivity": std_metrics["sensitivity"],
        "specificity": std_metrics["specificity"],
        "precision": std_metrics["precision"],
        "f1": std_metrics["f1"],
        "brier": std_metrics["brier"],
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "normalized_utility": best_ut,
        "median_warning_time": np.median(warning_times) if len(warning_times) > 0 else 0.0,
        "mean_warning_time": np.mean(warning_times) if len(warning_times) > 0 else 0.0,
        "false_alerts_per_100h": false_alert_rate,
        "alert_rate": overall_alert_rate
    }
    
    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(os.path.join(results_metrics_dir, "xgboost_metrics.csv"), index=False)
    logger.info("Saved xgboost_metrics.csv")
    
    # 9. Save all artifacts
    artifacts = {
        "model": best_xgb_model,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "best_threshold": best_thresh
    }
    
    with open(os.path.join(models_dir, "xgboost_model.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
    logger.info("Saved xgboost_model.pkl model artifact")

if __name__ == "__main__":
    train_xgboost()
