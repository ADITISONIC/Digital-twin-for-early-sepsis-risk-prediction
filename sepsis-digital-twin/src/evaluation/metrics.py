import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)

def calculate_physionet_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    """
    Computes raw utility for a single patient trajectory.
    Matches the official PhysioNet 2019 challenge formula.
    """
    labels = np.array(labels, dtype=int)
    predictions = np.array(predictions, dtype=int)
    
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions must have the same length.")
        
    n = len(labels)
    
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')
        
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal
    
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn
                
    return float(np.sum(u))

def compute_cohort_utility(cohort_labels, cohort_predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    """
    Computes normalized utility score across a list of patient trajectories.
    """
    num_files = len(cohort_labels)
    observed_utilities = np.zeros(num_files)
    best_utilities     = np.zeros(num_files)
    inaction_utilities = np.zeros(num_files)
    
    for k in range(num_files):
        labels = np.array(cohort_labels[k], dtype=int)
        num_rows = len(labels)
        observed_preds = np.array(cohort_predictions[k], dtype=int)
        
        best_preds = np.zeros(num_rows)
        inaction_preds = np.zeros(num_rows)
        
        if np.any(labels):
            t_sepsis = np.argmax(labels) - dt_optimal
            best_preds[max(0, int(t_sepsis + dt_early)) : min(int(t_sepsis + dt_late + 1), num_rows)] = 1
            
        observed_utilities[k] = calculate_physionet_utility(labels, observed_preds, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        best_utilities[k]     = calculate_physionet_utility(labels, best_preds, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        inaction_utilities[k] = calculate_physionet_utility(labels, inaction_preds, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        
    unnormalized_observed_utility = np.sum(observed_utilities)
    unnormalized_best_utility     = np.sum(best_utilities)
    unnormalized_inaction_utility = np.sum(inaction_utilities)
    
    denominator = unnormalized_best_utility - unnormalized_inaction_utility
    if denominator == 0:
        return 0.0, unnormalized_observed_utility
        
    normalized_observed_utility = (unnormalized_observed_utility - unnormalized_inaction_utility) / denominator
    return float(normalized_observed_utility), float(unnormalized_observed_utility)

def calculate_standard_metrics(y_true, y_prob, y_pred):
    """
    Computes standard classification metrics at the row/hour level.
    """
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    y_pred = np.array(y_pred, dtype=int)
    
    # AUROC
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except:
        auroc = 0.5
        
    # AUPRC
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        auprc = float(auc(recall_curve, precision_curve))
    except:
        auprc = 0.0
        
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "brier": brier,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }

def calculate_calibration_stats(y_true, y_prob):
    """
    Computes calibration slope and intercept.
    Uses logit-transformed probabilities to fit logistic regression.
    """
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    
    # Clip to avoid log(0) or log(1)
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    logits = np.log(y_prob / (1.0 - y_prob))
    
    try:
        lr = LogisticRegression()
        lr.fit(logits.reshape(-1, 1), y_true)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
    except Exception as e:
        logger.warning("Calibration slope fit failed: %s. Using default 1.0/0.0.", str(e))
        slope = 1.0
        intercept = 0.0
        
    return slope, intercept

def calculate_warning_time(cohort_labels, cohort_predictions):
    """
    For each sepsis patient, calculates warning time in hours relative to clinical onset.
    """
    dt_optimal = -6
    warning_times = []
    
    for labels, predictions in zip(cohort_labels, cohort_predictions):
        labels = np.array(labels)
        predictions = np.array(predictions)
        
        if np.any(labels == 1):
            # Clinical onset is the first index where SepsisLabel is 1 plus 6 hours (since labels are shifted 6h early)
            # Actually, the labels in the dataset are already shifted by 6h.
            # So the first index where label == 1 is the 6h early prediction onset!
            # The actual clinical onset hour is: t_onset = ArgMax(labels) + 6
            t_label_onset = np.argmax(labels)
            t_clinical_onset = t_label_onset + 6
            
            # Earliest model alert (prediction == 1)
            alerts = np.where(predictions == 1)[0]
            if len(alerts) > 0:
                first_alert = alerts[0]
                # Warning time is clinical onset minus first alert time
                # If alert occurs AFTER clinical onset, warning time is negative
                warn_time = t_clinical_onset - first_alert
                warning_times.append(warn_time)
                
    return warning_times

def calculate_alarm_burden(cohort_labels, cohort_predictions):
    """
    Computes false alerts per 100 patient-hours and overall alert rate.
    """
    total_hours = 0
    false_alert_hours = 0
    total_alerts = 0
    
    for labels, predictions in zip(cohort_labels, cohort_predictions):
        labels = np.array(labels)
        predictions = np.array(predictions)
        
        total_hours += len(labels)
        total_alerts += np.sum(predictions == 1)
        
        # A false alert hour is when prediction == 1 but SepsisLabel == 0
        false_alert_hours += np.sum((predictions == 1) & (labels == 0))
        
    false_alerts_per_100h = (false_alert_hours / total_hours) * 100 if total_hours > 0 else 0.0
    alert_rate = total_alerts / total_hours if total_hours > 0 else 0.0
    
    return false_alerts_per_100h, alert_rate

def bootstrap_metrics(cohort_labels, cohort_probs, cohort_preds, n_bootstraps=200, seed=42):
    """
    Patient-level bootstrap resampling to compute 95% Confidence Intervals.
    """
    np.random.seed(seed)
    n_patients = len(cohort_labels)
    
    boot_auroc = []
    boot_auprc = []
    boot_utility = []
    boot_sensitivity = []
    boot_specificity = []
    boot_warning_median = []
    
    logger.info("Running bootstrap validation...")
    for i in range(n_bootstraps):
        # Sample patient indices with replacement
        indices = np.random.choice(n_patients, size=n_patients, replace=True)
        
        boot_lbl = [cohort_labels[idx] for idx in indices]
        boot_prb = [cohort_probs[idx] for idx in indices]
        boot_prd = [cohort_preds[idx] for idx in indices]
        
        # Flatten for standard metrics
        lbl_flat = np.concatenate(boot_lbl)
        prb_flat = np.concatenate(boot_prb)
        prd_flat = np.concatenate(boot_prd)
        
        std_mets = calculate_standard_metrics(lbl_flat, prb_flat, prd_flat)
        norm_ut, _ = compute_cohort_utility(boot_lbl, boot_prd)
        warns = calculate_warning_time(boot_lbl, boot_prd)
        
        boot_auroc.append(std_mets["auroc"])
        boot_auprc.append(std_mets["auprc"])
        boot_utility.append(norm_ut)
        boot_sensitivity.append(std_mets["sensitivity"])
        boot_specificity.append(std_mets["specificity"])
        if len(warns) > 0:
            boot_warning_median.append(np.median(warns))
        else:
            boot_warning_median.append(0.0)
            
    # Calculate percentiles
    def get_ci(vals):
        vals = sorted(vals)
        low = np.percentile(vals, 2.5)
        high = np.percentile(vals, 97.5)
        mean_val = np.mean(vals)
        return mean_val, low, high
        
    return {
        "auroc": get_ci(boot_auroc),
        "auprc": get_ci(boot_auprc),
        "utility": get_ci(boot_utility),
        "sensitivity": get_ci(boot_sensitivity),
        "specificity": get_ci(boot_specificity),
        "warning_time": get_ci(boot_warning_median)
    }
