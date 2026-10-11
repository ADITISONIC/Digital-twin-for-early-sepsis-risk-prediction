import os
import pandas as pd
import numpy as np
import pickle
import logging
import gc
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from src.utils.helpers import load_config, set_seeds
from src.data.loader import load_cohort_data
from src.features.build_features import build_dynamic_features_pipeline
from src.models.train_lstm import SepsisLSTM
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

def predict_lstm_batched(model, X_list, batch_size=64):
    """
    Predicts Sepsis probabilities in batches using padding.
    Retains only the original length of each patient's sequence.
    """
    model.eval()
    all_probs = []
    n_samples = len(X_list)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_x_list = X_list[i : i + batch_size]
            lengths = [len(x) for x in batch_x_list]
            
            # Convert numpy arrays to tensors
            tensors = [torch.tensor(x, dtype=torch.float32) for x in batch_x_list]
            
            # Pad sequences within the batch
            # Shape after padding: (batch_size, max_seq_len_in_batch, input_dim)
            x_padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0)
            
            # Forward pass
            preds_padded = model(x_padded)  # Shape: (batch_size, max_seq_len_in_batch, 1)
            
            # Extract actual predictions using original sequence lengths
            for j, length in enumerate(lengths):
                probs = preds_padded[j, :length, 0].cpu().numpy()
                all_probs.append(probs)
                
            if (i + batch_size) % (batch_size * 20) == 0 or i + batch_size >= n_samples:
                logger.info("LSTM prediction progress: %d / %d patients", min(i + batch_size, n_samples), n_samples)
                
    return all_probs

def run_smoke_test(model, preprocessor, scaler, feature_cols, config, base_dir):
    """
    Performs a validation check on 5 patients from Hospital B to ensure correctness of batched sequence prediction.
    """
    logger.info("Running LSTM prediction smoke test on 5 patients...")
    splits_dir = os.path.join(base_dir, config["paths"]["splits_dir"])
    holdout_ids = pd.read_csv(os.path.join(splits_dir, "holdout_hospitalB_patients.csv"))["PatientID"].tolist()
    smoke_ids = holdout_ids[:5]
    
    test_dfs = load_cohort_data(smoke_ids, 'B', config)
    test_dfs = preprocessor.transform(test_dfs)
    test_dfs = build_dynamic_features_pipeline(
        test_dfs,
        vitals=config["variables"]["vitals"],
        labs=config["variables"]["labs"]
    )
    
    X_test_stacked = np.vstack([df[feature_cols].values for df in test_dfs])
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test_stacked)
    else:
        X_test_scaled = X_test_stacked
        
    lengths = [len(df) for df in test_dfs]
    indices = np.cumsum([0] + lengths)
    X_list = [X_test_scaled[indices[i]:indices[i+1]] for i in range(len(lengths))]
    
    # Run predictions in tiny batches of 2
    probs = predict_lstm_batched(model, X_list, batch_size=2)
    
    assert len(probs) == len(X_list), f"Expected {len(X_list)} outputs, got {len(probs)}"
    for idx, (p, x) in enumerate(zip(probs, X_list)):
        assert len(p) == len(x), f"Patient {idx} output sequence length mismatch: expected {len(x)}, got {len(p)}"
        
    logger.info("Smoke test passed successfully!")

def evaluate_lstm_holdout():
    config = load_config()
    set_seeds(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    splits_dir = os.path.join(base_dir, config["paths"]["splits_dir"])
    models_dir = os.path.join(base_dir, config["paths"]["models_dir"])
    results_dir = os.path.join(base_dir, config["paths"]["results_dir"])
    results_metrics_dir = os.path.join(results_dir, "metrics")
    results_figures_dir = os.path.join(results_dir, "figures")
    
    os.makedirs(results_metrics_dir, exist_ok=True)
    os.makedirs(results_figures_dir, exist_ok=True)
    
    # 1. Load LSTM metadata and verify
    metadata_path = os.path.join(models_dir, "lstm_metadata.pkl")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"LSTM metadata not found at: {metadata_path}")
        
    logger.info("Loading preprocessor, scaler, and features from LSTM metadata...")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
        
    preprocessor = metadata["preprocessor"]
    scaler = metadata.get("scaler", None)
    feature_cols = metadata["feature_cols"]
    threshold = metadata["best_threshold"]
    
    # 2. Reconstruct LSTM model and verify weights exist
    model_path = os.path.join(models_dir, "lstm_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model weights not found at: {model_path}")
        
    input_dim = len(feature_cols)
    model = SepsisLSTM(input_dim, 32)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 3. Run smoke test on 5 patients
    run_smoke_test(model, preprocessor, scaler, feature_cols, config, base_dir)
    
    # 4. Load and process Hospital B test set (20,000 patients)
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
    
    test_patient_labels = [df[config["data"]["target_col"]].values for df in test_dfs]
    lengths_test = [len(df) for df in test_dfs]
    indices_test = np.cumsum([0] + lengths_test)
    
    X_test = np.vstack([df[feature_cols].values for df in test_dfs])
    y_test = np.concatenate([df[config["data"]["target_col"]].values for df in test_dfs])
    
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
        
    X_test_list = [X_test_scaled[indices_test[i]:indices_test[i+1]] for i in range(len(lengths_test))]
    
    del test_dfs
    gc.collect()
    
    # 5. Predict Test cohort in batches
    logger.info("Predicting LSTM on Hospital B test set in batches...")
    cohort_probs_test = predict_lstm_batched(model, X_test_list, batch_size=64)
    
    test_probs = np.concatenate(cohort_probs_test)
    test_preds_flat = (test_probs >= threshold).astype(int)
    cohort_preds_test = [(probs >= threshold).astype(int) for probs in cohort_probs_test]
    
    # 6. Calculate metrics on Hospital B
    logger.info("Calculating LSTM standard metrics on Hospital B...")
    std_metrics = calculate_standard_metrics(y_test, test_probs, test_preds_flat)
    slope, intercept = calculate_calibration_stats(y_test, test_probs)
    warning_times = calculate_warning_time(test_patient_labels, cohort_preds_test)
    false_alert_rate, overall_alert_rate = calculate_alarm_burden(test_patient_labels, cohort_preds_test)
    norm_ut, raw_ut = compute_cohort_utility(test_patient_labels, cohort_preds_test)
    
    # 7. Run bootstrap for confidence intervals on Hospital B (200 iterations)
    logger.info("Running bootstrap confidence intervals for LSTM on Hospital B...")
    boot_results = bootstrap_metrics(
        test_patient_labels, cohort_probs_test, cohort_preds_test, n_bootstraps=200, seed=config["random_seed"]
    )
    
    # Save holdout metrics
    holdout_summary = {
        "model": "lstm",
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
    
    holdout_df = pd.DataFrame([holdout_summary])
    holdout_df.to_csv(os.path.join(results_metrics_dir, "lstm_holdout_metrics.csv"), index=False)
    logger.info("Saved lstm_holdout_metrics.csv")
    
    # 8. Load and process Hospital A validation set (4,068 patients)
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
    
    if scaler is not None:
        X_val_scaled = scaler.transform(X_val)
    else:
        X_val_scaled = X_val
        
    X_val_list = [X_val_scaled[indices_val[i]:indices_val[i+1]] for i in range(len(lengths_val))]
    
    del val_dfs, X_test, X_val, X_test_scaled, X_val_scaled
    gc.collect()
    
    # 9. Predict Validation cohort in batches
    logger.info("Predicting LSTM on Hospital A validation set in batches...")
    cohort_probs_val = predict_lstm_batched(model, X_val_list, batch_size=64)
    val_probs = np.concatenate(cohort_probs_val)
    
    # 10. Generate and save calibration curve comparison figure
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
    plt.title("Reliability Curve: LSTM Validation vs Test")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(results_figures_dir, "calibration_lstm.png"), dpi=300)
    plt.close()
    logger.info("Saved calibration curve figure results/figures/calibration_lstm.png")
    
    # 11. Final Summary output
    print("\n" + "="*50)
    print("LSTM HOLDOUT EVALUATION COMPLETE")
    print("="*50)
    for k, v in holdout_summary.items():
        print(f"{k:30}: {v}")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate_lstm_holdout()
