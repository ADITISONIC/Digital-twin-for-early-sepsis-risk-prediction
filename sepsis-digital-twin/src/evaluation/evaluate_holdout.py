import os
import pandas as pd
import numpy as np
import pickle
import logging
import gc
import json
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from src.utils.helpers import load_config, set_seeds
from src.data.loader import load_cohort_data
from src.features.build_features import build_dynamic_features_pipeline
from src.evaluation.metrics import (
    calculate_standard_metrics,
    compute_cohort_utility,
    calculate_calibration_stats,
    calculate_warning_time,
    calculate_alarm_burden,
    bootstrap_metrics
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_on_holdout():
    config = load_config()
    set_seeds(config["random_seed"])
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    splits_dir = os.path.join(base_dir, config["paths"]["splits_dir"])
    models_dir = os.path.join(base_dir, config["paths"]["models_dir"])
    results_dir = os.path.join(base_dir, config["paths"]["results_dir"])
    results_metrics_dir = os.path.join(results_dir, "metrics")
    results_figures_dir = os.path.join(results_dir, "figures")
    
    os.makedirs(results_metrics_dir, exist_ok=True)
    os.makedirs(results_figures_dir, exist_ok=True)
    
    # 1. Load CausalPreprocessor and features from XGBoost model artifact (frozen)
    logger.info("Loading preprocessor and feature list from XGBoost model artifact...")
    xgb_artifact_path = os.path.join(models_dir, "xgboost_model.pkl")
    with open(xgb_artifact_path, "rb") as f:
        xgb_artifact = pickle.load(f)
    preprocessor = xgb_artifact["preprocessor"]
    feature_cols = xgb_artifact["feature_cols"]
    del xgb_artifact
    gc.collect()
    
    # 2. Load and process Hospital B test set (20,000 patients) ONCE
    holdout_ids = pd.read_csv(os.path.join(splits_dir, "holdout_hospitalB_patients.csv"))["PatientID"].tolist()
    logger.info("Loaded Hospital B patient IDs: %d", len(holdout_ids))
    
    test_dfs = load_cohort_data(holdout_ids, 'B', config)
    test_dfs = preprocessor.transform(test_dfs)
    gc.collect()
    
    test_dfs = build_dynamic_features_pipeline(
        test_dfs,
        vitals=config["variables"]["vitals"],
        labs=config["variables"]["labs"]
    )
    gc.collect()
    
    # Reconstruct patient-level predictions and labels for evaluation
    test_patient_labels = [df[config["data"]["target_col"]].values for df in test_dfs]
    lengths_test = [len(df) for df in test_dfs]
    indices_test = np.cumsum([0] + lengths_test)
    
    X_test = np.vstack([df[feature_cols].values for df in test_dfs])
    y_test = np.concatenate([df[config["data"]["target_col"]].values for df in test_dfs])
    
    del test_dfs
    gc.collect()
    
    # 3. Load and process Hospital A validation set (4,068 patients) ONCE for calibration comparison
    val_ids = pd.read_csv(os.path.join(splits_dir, "validation_patients.csv"))["PatientID"].tolist()
    logger.info("Loaded Hospital A validation patient IDs: %d", len(val_ids))
    
    val_dfs = load_cohort_data(val_ids, 'A', config)
    val_dfs = preprocessor.transform(val_dfs)
    gc.collect()
    
    val_dfs = build_dynamic_features_pipeline(
        val_dfs,
        vitals=config["variables"]["vitals"],
        labs=config["variables"]["labs"]
    )
    gc.collect()
    
    lengths_val = [len(df) for df in val_dfs]
    indices_val = np.cumsum([0] + lengths_val)
    
    X_val = np.vstack([df[feature_cols].values for df in val_dfs])
    y_val = np.concatenate([df[config["data"]["target_col"]].values for df in val_dfs])
    
    del val_dfs
    gc.collect()
    
    # 4. Evaluate models
    models_to_evaluate = ["logistic_regression", "xgboost", "lstm"]
    
    for model_name in models_to_evaluate:
        logger.info("Evaluating model: %s on Hospital B...", model_name)
        
        if model_name == "lstm":
            import torch
            from src.models.train_lstm import SepsisLSTM
            
            # Load PyTorch model and metadata
            artifact_path = os.path.join(models_dir, "lstm_metadata.pkl")
            with open(artifact_path, "rb") as f:
                artifact = pickle.load(f)
            threshold = artifact["best_threshold"]
            scaler = artifact.get("scaler", None)
            feature_cols_metadata = artifact["feature_cols"]
            del artifact
            gc.collect()
            
            # Re-initialize PyTorch LSTM model
            input_dim = len(feature_cols_metadata)
            model = SepsisLSTM(input_dim, 32)
            model.load_state_dict(torch.load(os.path.join(models_dir, "lstm_model.pt")))
            model.eval()
            
            # Scale
            if scaler is not None:
                X_test_scaled = scaler.transform(X_test)
                X_val_scaled = scaler.transform(X_val)
            else:
                X_test_scaled = X_test
                X_val_scaled = X_val
                
            # Reconstruct list and predict for test (patient by patient)
            X_test_list = [X_test_scaled[indices_test[i]:indices_test[i+1]] for i in range(len(lengths_test))]
            cohort_probs_test = []
            
            logger.info("Predicting LSTM on Hospital B test set...")
            with torch.no_grad():
                for x in X_test_list:
                    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0) # (1, seq_len, d)
                    pred = model(x_tensor)
                    cohort_probs_test.append(pred[0, :, 0].numpy())
                    
            test_probs = np.concatenate(cohort_probs_test)
            test_preds_flat = (test_probs >= threshold).astype(int)
            cohort_preds_test = [(probs >= threshold).astype(int) for probs in cohort_probs_test]
            
            # Reconstruct list and predict for validation
            X_val_list = [X_val_scaled[indices_val[i]:indices_val[i+1]] for i in range(len(lengths_val))]
            cohort_probs_val = []
            with torch.no_grad():
                for x in X_val_list:
                    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                    pred = model(x_tensor)
                    cohort_probs_val.append(pred[0, :, 0].numpy())
                    
            val_probs = np.concatenate(cohort_probs_val)
            
            # Free memory
            del X_test_scaled, X_val_scaled, X_test_list, X_val_list
            gc.collect()
            
        else:
            artifact_path = os.path.join(models_dir, f"{model_name}.pkl" if model_name == "logistic_regression" else "xgboost_model.pkl")
            with open(artifact_path, "rb") as f:
                artifact = pickle.load(f)
                
            model = artifact["model"]
            threshold = artifact["best_threshold"]
            scaler = artifact.get("scaler", None)
            del artifact
            gc.collect()
            
            # Scale test and validation datasets if scaler exists
            if scaler is not None:
                X_test_scaled = scaler.transform(X_test)
                X_val_scaled = scaler.transform(X_val)
            else:
                X_test_scaled = X_test
                X_val_scaled = X_val
                
            # Predict test set probabilities
            test_probs = model.predict_proba(X_test_scaled)[:, 1]
            test_preds_flat = (test_probs >= threshold).astype(int)
            
            cohort_preds_test = [test_preds_flat[indices_test[i]:indices_test[i+1]] for i in range(len(lengths_test))]
            cohort_probs_test = [test_probs[indices_test[i]:indices_test[i+1]] for i in range(len(lengths_test))]
            
            # Predict validation set probabilities
            val_probs = model.predict_proba(X_val_scaled)[:, 1]
            
            # Free scaled matrices to keep memory low
            if scaler is not None:
                del X_test_scaled
                del X_val_scaled
                gc.collect()
            
        # Calculate performance metrics on Hospital B
        std_metrics = calculate_standard_metrics(y_test, test_probs, test_preds_flat)
        slope, intercept = calculate_calibration_stats(y_test, test_probs)
        warning_times = calculate_warning_time(test_patient_labels, cohort_preds_test)
        false_alert_rate, overall_alert_rate = calculate_alarm_burden(test_patient_labels, cohort_preds_test)
        norm_ut, raw_ut = compute_cohort_utility(test_patient_labels, cohort_preds_test)
        
        # Run bootstrap for confidence intervals on Hospital B
        logger.info("Running bootstrap confidence intervals for %s on Hospital B...", model_name)
        boot_results = bootstrap_metrics(
            test_patient_labels, cohort_probs_test, cohort_preds_test, n_bootstraps=200, seed=config["random_seed"]
        )
        
        # Log holdout metrics
        holdout_summary = {
            "model": model_name,
            "threshold": threshold,
            "auroc": std_metrics["auroc"],
            "auroc_95ci": f"({boot_results['auroc'][1]:.4f}, {boot_results['auroc'][2]:.4f})",
            "auprc": std_metrics["auprc"],
            "auprc_95ci": f"({boot_results['auprc'][1]:.4f}, {boot_results['auprc'][2]:.4f})",
            "sensitivity": std_metrics["sensitivity"],
            "sensitivity_95ci": f"({boot_results['sensitivity'][1]:.4f}, {boot_results['sensitivity'][2]:.4f})",
            "specificity": std_metrics["specificity"],
            "specificity_95ci": f"({boot_results['specificity'][1]:.4f}, {boot_results['specificity'][2]:.4f})",
            "precision": std_metrics["precision"],
            "f1": std_metrics["f1"],
            "brier": std_metrics["brier"],
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "normalized_utility": norm_ut,
            "utility_95ci": f"({boot_results['utility'][1]:.4f}, {boot_results['utility'][2]:.4f})",
            "median_warning_time": np.median(warning_times) if len(warning_times) > 0 else 0.0,
            "warning_time_95ci": f"({boot_results['warning_time'][1]:.1f}, {boot_results['warning_time'][2]:.1f})",
            "mean_warning_time": np.mean(warning_times) if len(warning_times) > 0 else 0.0,
            "false_alerts_per_100h": false_alert_rate,
            "alert_rate": overall_alert_rate
        }
        
        # Save holdout metrics
        holdout_df = pd.DataFrame([holdout_summary])
        holdout_df.to_csv(os.path.join(results_metrics_dir, f"{model_name}_holdout_metrics.csv"), index=False)
        logger.info("Saved %s_holdout_metrics.csv", model_name)
        
        # Plot calibration curve comparison
        prob_true_val, prob_pred_val = calibration_curve(y_val, val_probs, n_bins=10)
        prob_true_test, prob_pred_test = calibration_curve(y_test, test_probs, n_bins=10)
        
        val_slope = calculate_calibration_stats(y_val, val_probs)[0]
        
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
        plt.plot(prob_pred_val, prob_true_val, "s-", label=f"Hospital A Validation (Slope: {val_slope:.2f})")
        plt.plot(prob_pred_test, prob_true_test, "o-", label=f"Hospital B Held-out (Slope: {slope:.2f})")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.ylim([-0.05, 1.05])
        plt.title(f"Reliability Curve: {model_name.upper()} Validation vs Test")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        
        # Filename based on model name
        fig_name = f"calibration_{'lr' if model_name == 'logistic_regression' else ('xgb' if model_name == 'xgboost' else 'lstm')}.png"
        plt.savefig(os.path.join(results_figures_dir, fig_name), dpi=300)
        plt.close()
        logger.info("Saved calibration curve figure for %s", model_name)
        
        gc.collect()

if __name__ == "__main__":
    evaluate_on_holdout()
