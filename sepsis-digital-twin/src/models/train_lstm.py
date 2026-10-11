import os
import pandas as pd
import numpy as np
import pickle
import logging
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

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

class SepsisLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(SepsisLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return torch.sigmoid(out)

class MaskedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super(MaskedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, preds, targets, mask):
        # preds, targets: (batch_size, seq_len, 1)
        # mask: (batch_size, seq_len, 1) boolean mask
        loss = - (self.pos_weight * targets * torch.log(preds + 1e-8) + (1.0 - targets) * torch.log(1.0 - preds + 1e-8))
        masked_loss = loss * mask.float()
        return masked_loss.sum() / torch.clamp(mask.sum(), min=1e-8)

def prepare_sequential_data(dfs, feature_cols, target_col, scaler=None, fit_scaler=False):
    """
    Prepares features and labels.
    """
    X_list = [df[feature_cols].values for df in dfs]
    y_list = [df[target_col].values for df in dfs]
    
    # Scale features
    if fit_scaler:
        scaler = StandardScaler()
        stacked = np.vstack(X_list)
        scaler.fit(stacked)
        
    if scaler is not None:
        X_list = [scaler.transform(x) for x in X_list]
        
    return X_list, y_list, scaler

def train_lstm():
    config = load_config()
    set_seeds(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    splits_dir = os.path.join(base_dir, config["paths"]["splits_dir"])
    models_dir = os.path.join(base_dir, config["paths"]["models_dir"])
    results_dir = os.path.join(base_dir, config["paths"]["results_dir"])
    results_metrics_dir = os.path.join(results_dir, "metrics")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_metrics_dir, exist_ok=True)
    
    # 1. Load patient ID lists
    train_ids = pd.read_csv(os.path.join(splits_dir, "train_patients.csv"))["PatientID"].tolist()
    val_ids = pd.read_csv(os.path.join(splits_dir, "validation_patients.csv"))["PatientID"].tolist()
    
    logger.info("Loaded patient IDs: Train=%d, Val=%d", len(train_ids), len(val_ids))
    
    # 2. Load and process training cohort data (limit=1500 for fast CPU training)
    train_dfs = load_cohort_data(train_ids, 'A', config, limit=1500)
    
    preprocessor = CausalPreprocessor(
        vitals=config["variables"]["vitals"],
        labs=config["variables"]["labs"],
        demographics=config["variables"]["demographics"],
        max_forward_fill_gap=config["preprocessing"]["max_forward_fill_gap"]
    )
    preprocessor.fit(train_dfs)
    
    train_dfs = preprocessor.transform(train_dfs)
    gc.collect()
    
    train_dfs = build_dynamic_features_pipeline(
        train_dfs,
        vitals=config["variables"]["vitals"],
        labs=config["variables"]["labs"]
    )
    gc.collect()
    
    exclude_cols = ["PatientID", "HospitalID", "SepsisLabel", "Unit1", "Unit2"]
    sample_df = train_dfs[0]
    feature_cols = [c for c in sample_df.columns if c not in exclude_cols]
    
    X_train_list, y_train_list, scaler = prepare_sequential_data(
        train_dfs, feature_cols, config["data"]["target_col"], fit_scaler=True
    )
    
    del train_dfs
    gc.collect()
    
    # 3. Load and process validation cohort data (limit=2000)
    val_dfs = load_cohort_data(val_ids, 'A', config, limit=2000)
    val_dfs = preprocessor.transform(val_dfs)
    gc.collect()
    
    val_dfs = build_dynamic_features_pipeline(
        val_dfs,
        vitals=config["variables"]["vitals"],
        labs=config["variables"]["labs"]
    )
    gc.collect()
    
    val_patient_labels = [df[config["data"]["target_col"]].values for df in val_dfs]
    
    X_val_list, y_val_list, _ = prepare_sequential_data(
        val_dfs, feature_cols, config["data"]["target_col"], scaler=scaler, fit_scaler=False
    )
    
    del val_dfs
    gc.collect()
    
    # 4. Class balance weight
    y_train_flat = np.concatenate(y_train_list)
    neg_count = np.sum(y_train_flat == 0)
    pos_count = np.sum(y_train_flat == 1)
    imbalance_ratio = float(neg_count / pos_count) if pos_count > 0 else 1.0
    logger.info("Class imbalance ratio: %.2f", imbalance_ratio)
    
    # 5. Build PyTorch model and optimizer
    input_dim = len(feature_cols)
    hidden_dim = 32
    model = SepsisLSTM(input_dim, hidden_dim)
    criterion = MaskedBCELoss(pos_weight=imbalance_ratio)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Pad training data to length 336 for batched training
    # PyTorch requires tensors of same shape for batching
    max_len = 336
    X_padded = []
    y_padded = []
    mask_padded = []
    
    for x, y in zip(X_train_list, y_train_list):
        seq_len = len(x)
        x_pad = np.zeros((max_len, input_dim))
        x_pad[:seq_len, :] = x
        
        y_pad = np.zeros((max_len, 1))
        y_pad[:seq_len, 0] = y
        
        mask = np.zeros((max_len, 1), dtype=bool)
        mask[:seq_len, 0] = True
        
        X_padded.append(x_pad)
        y_padded.append(y_pad)
        mask_padded.append(mask)
        
    X_train_tensor = torch.tensor(np.array(X_padded), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_padded), dtype=torch.float32)
    mask_train_tensor = torch.tensor(np.array(mask_padded), dtype=torch.bool)
    
    # Simple batched training loop
    epochs = 4
    batch_size = 64
    n_samples = len(X_train_list)
    
    logger.info("Training PyTorch LSTM...")
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, n_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]
            batch_mask = mask_train_tensor[indices]
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y, batch_mask)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        logger.info("Epoch %d/%d - Loss: %.4f", epoch+1, epochs, epoch_loss / batches)
        
    # 6. Evaluate on validation set and optimize threshold
    model.eval()
    logger.info("Evaluating LSTM on validation set...")
    
    val_probs_list = []
    with torch.no_grad():
        for x in X_val_list:
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0) # batch size 1
            pred = model(x_tensor)
            val_probs_list.append(pred[0, :, 0].numpy())
            
    val_probs_flat = np.concatenate(val_probs_list)
    y_val_flat = np.concatenate(val_patient_labels)
    
    # Optimize threshold on validation utility
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_ut = -999.0
    
    for th in thresholds:
        cohort_preds = [(p >= th).astype(int) for p in val_probs_list]
        ut, _ = compute_cohort_utility(val_patient_labels, cohort_preds)
        if ut > best_ut:
            best_ut = ut
            best_thresh = th
            
    logger.info("Best validation utility: %.4f at threshold %.2f", best_ut, best_thresh)
    
    val_preds_flat = (val_probs_flat >= best_thresh).astype(int)
    cohort_preds = [(p >= best_thresh).astype(int) for p in val_probs_list]
    
    # Compute metrics
    std_metrics = calculate_standard_metrics(y_val_flat, val_probs_flat, val_preds_flat)
    slope, intercept = calculate_calibration_stats(y_val_flat, val_probs_flat)
    warning_times = calculate_warning_time(val_patient_labels, cohort_preds)
    false_alert_rate, overall_alert_rate = calculate_alarm_burden(val_patient_labels, cohort_preds)
    
    metrics_summary = {
        "model": "lstm",
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
    
    # Save validation metrics
    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(os.path.join(results_metrics_dir, "lstm_metrics.csv"), index=False)
    logger.info("Saved lstm_metrics.csv")
    
    # Save PyTorch model state dict
    torch.save(model.state_dict(), os.path.join(models_dir, "lstm_model.pt"))
    
    artifacts = {
        "preprocessor": preprocessor,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "best_threshold": best_thresh
    }
    
    with open(os.path.join(models_dir, "lstm_metadata.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
        
    logger.info("Saved LSTM PyTorch artifacts successfully")

if __name__ == "__main__":
    train_lstm()
