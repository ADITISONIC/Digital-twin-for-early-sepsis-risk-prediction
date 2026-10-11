import os
import pandas as pd
import numpy as np
import pickle
import logging
import gc
import shap
import matplotlib.pyplot as plt

from src.utils.helpers import load_config, set_seeds
from src.data.loader import load_cohort_data
from src.features.build_features import build_dynamic_features_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_shap_and_cases():
    config = load_config()
    set_seeds(config["random_seed"])
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    splits_dir = os.path.join(base_dir, config["paths"]["splits_dir"])
    models_dir = os.path.join(base_dir, config["paths"]["models_dir"])
    results_dir = os.path.join(base_dir, config["paths"]["results_dir"])
    results_figures_dir = os.path.join(results_dir, "figures")
    case_studies_dir = os.path.join(results_dir, "case_studies")
    
    os.makedirs(results_figures_dir, exist_ok=True)
    os.makedirs(case_studies_dir, exist_ok=True)
    
    # 1. Load model artifact
    logger.info("Loading XGBoost model artifact for SHAP explainability...")
    with open(os.path.join(models_dir, "xgboost_model.pkl"), "rb") as f:
        artifact = pickle.load(f)
        
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    feature_cols = artifact["feature_cols"]
    threshold = artifact["best_threshold"]
    del artifact
    gc.collect()
    
    # 2. Load validation data to calculate SHAP
    val_ids = pd.read_csv(os.path.join(splits_dir, "validation_patients.csv"))["PatientID"].tolist()
    
    # Load validation data (sample first 1000 for faster SHAP calculation)
    val_dfs = load_cohort_data(val_ids[:1000], 'A', config)
    val_dfs = preprocessor.transform(val_dfs)
    gc.collect()
    
    val_dfs = build_dynamic_features_pipeline(
        val_dfs,
        vitals=config["variables"]["vitals"],
        labs=config["variables"]["labs"]
    )
    gc.collect()
    
    val_modeling_df = pd.concat(val_dfs, axis=0, ignore_index=True)
    X_val = val_modeling_df[feature_cols]
    
    # Sample 2000 rows for SHAP background/explainer (SHAP is computationally intensive)
    X_sample = shap.sample(X_val, 2000, random_state=config["random_seed"])
    
    logger.info("Computing SHAP values for XGBoost model...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # Generate SHAP global summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Global Summary: Feature Attributions for Sepsis Risk", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_figures_dir, "shap_summary.png"), dpi=300)
    plt.close()
    logger.info("Saved shap_summary.png")
    
    # Generate SHAP bar plot for feature importance
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("SHAP Global Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_figures_dir, "shap_importance.png"), dpi=300)
    plt.close()
    logger.info("Saved shap_importance.png")
    
    del X_val
    del X_sample
    del val_modeling_df
    gc.collect()
    
    # 3. Identify Patient Case Studies (TP, FP, FN)
    logger.info("Identifying candidate case studies...")
    
    tp_id, fp_id, fn_id = None, None, None
    
    for df in val_dfs:
        labels = df[config["data"]["target_col"]].values
        probs = model.predict_proba(df[feature_cols].values)[:, 1]
        preds = (probs >= threshold).astype(int)
        
        has_sepsis = np.any(labels == 1)
        has_alert = np.any(preds == 1)
        p_id = df["PatientID"].iloc[0]
        
        # True Positive: develops sepsis, and alerted before/at onset
        if has_sepsis and has_alert and tp_id is None:
            # Check if alert is before or shortly after onset
            onset_idx = np.argmax(labels)
            alert_idx = np.where(preds == 1)[0][0]
            if alert_idx <= onset_idx + 3:
                tp_id = p_id
                
        # False Positive: no sepsis, but alerted
        if not has_sepsis and has_alert and fp_id is None:
            if len(df) > 10:
                fp_id = p_id
                
        # False Negative: develops sepsis, but never alerted
        if has_sepsis and not has_alert and fn_id is None:
            fn_id = p_id
            
        if tp_id and fp_id and fn_id:
            break
            
    logger.info("Case Studies Selected: TP=%s, FP=%s, FN=%s", tp_id, fp_id, fn_id)
    
    del val_dfs
    gc.collect()
    
    # Helper to plot timeline
    cases = {"tp": tp_id, "fp": fp_id, "fn": fn_id}
    for case_type, p_id in cases.items():
        if p_id is None:
            logger.warning("No candidate found for case: %s", case_type)
            continue
            
        # Get patient data
        pat_dfs = load_cohort_data([p_id], 'A', config)
        pat_preprocessed = preprocessor.transform(pat_dfs)
        pat_feat_dfs = build_dynamic_features_pipeline(
            pat_preprocessed,
            vitals=config["variables"]["vitals"],
            labs=config["variables"]["labs"]
        )
        pat_df = pat_feat_dfs[0]
        
        probs = model.predict_proba(pat_df[feature_cols].values)[:, 1]
        labels = pat_df[config["data"]["target_col"]].values
        hours = pat_df["ICULOS"].values
        
        # Create timeline plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Risk Probability plot
        ax1.plot(hours, probs, "b-", linewidth=2, label="Model Sepsis Risk (Probability)")
        ax1.axhline(threshold, color="red", linestyle="--", label=f"Decision Alert Threshold ({threshold:.2f})")
        ax1.fill_between(hours, 0, labels, color="orange", alpha=0.2, label="True Sepsis Period (Ground Truth)")
        ax1.set_ylabel("Risk Probability", fontsize=12)
        ax1.set_ylim([-0.05, 1.05])
        ax1.set_title(f"Case Study: {case_type.upper()} Timeline (Patient ID: {p_id})", fontsize=14)
        ax1.legend(loc="upper left")
        ax1.grid(True)
        
        # Trajectories plot (Key vitals: Heart Rate and Temp)
        hr_vals = pat_df["HR"].values
        temp_vals = pat_df["Temp"].values
        
        ax2.plot(hours, hr_vals, "g-", label="Heart Rate (bpm)")
        ax2.set_ylabel("Heart Rate (bpm)", color="g", fontsize=12)
        ax2.tick_params(axis='y', labelcolor='g')
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(hours, temp_vals, "m-", label="Temperature (C)")
        ax2_twin.set_ylabel("Temperature (C)", color="m", fontsize=12)
        ax2_twin.tick_params(axis='y', labelcolor='m')
        
        ax2.set_xlabel("ICU Stay Hour (ICULOS)", fontsize=12)
        ax2.grid(True)
        
        # Merge legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        
        plt.tight_layout()
        plt.savefig(os.path.join(case_studies_dir, f"case_{case_type}.png"), dpi=300)
        plt.close()
        logger.info("Saved case study plot case_%s.png", case_type)
        
        del pat_dfs
        del pat_feat_dfs
        gc.collect()

if __name__ == "__main__":
    run_shap_and_cases()
