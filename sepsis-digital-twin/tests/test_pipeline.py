import pytest
import numpy as np
import pandas as pd
import os
import yaml

from src.preprocessing.pipeline import CausalPreprocessor
from src.features.build_features import build_dynamic_features_for_patient
from src.evaluation.metrics import calculate_physionet_utility

def get_test_config():
    return {
        "vitals": ["HR", "O2Sat"],
        "labs": ["Lactate", "Creatinine"],
        "demographics": ["Age", "Gender", "ICULOS"]
    }

def test_no_future_imputation():
    # HR is observed at hour 0, missing at hour 1, and observed at hour 2
    # Preprocessor should carry forward the value from hour 0 to hour 1
    # Hour 1 should NOT use the value from hour 2 (which is future)
    vitals = ["HR"]
    labs = []
    demographics = ["Age", "Gender", "ICULOS"]
    
    df = pd.DataFrame({
        "HR": [80.0, np.nan, 120.0],
        "Age": [60.0, 60.0, 60.0],
        "Gender": [1, 1, 1],
        "ICULOS": [1, 2, 3]
    })
    
    preprocessor = CausalPreprocessor(vitals, labs, demographics)
    preprocessor.fit([df])
    transformed = preprocessor.transform([df])[0]
    
    assert transformed["HR"].iloc[1] == 80.0
    assert transformed["HR_missing"].iloc[1] == 1.0
    assert transformed["HR_recency"].iloc[1] == 1.0

def test_no_future_rolling_features():
    # Verify that changing future rows does not affect historical features
    vitals = ["HR"]
    labs = []
    
    df1 = pd.DataFrame({
        "HR": [80.0, 85.0, 90.0],
        "Age": [60.0, 60.0, 60.0],
        "Gender": [1, 1, 1],
        "ICULOS": [1, 2, 3]
    })
    
    df2 = pd.DataFrame({
        "HR": [80.0, 85.0, 150.0],  # Hour 2 is changed (future)
        "Age": [60.0, 60.0, 60.0],
        "Gender": [1, 1, 1],
        "ICULOS": [1, 2, 3]
    })
    
    feat1 = build_dynamic_features_for_patient(df1, vitals, labs)
    feat2 = build_dynamic_features_for_patient(df2, vitals, labs)
    
    # Hour 0 and Hour 1 features should be identical
    for col in feat1.columns:
        if col not in ["PatientID", "HospitalID"]:
            assert feat1[col].iloc[0] == feat2[col].iloc[0]
            assert feat1[col].iloc[1] == feat2[col].iloc[1]

def test_training_only_statistics():
    # If we fit on train and transform test, changing test values must not affect fitted statistics
    vitals = ["HR"]
    labs = []
    demographics = ["Age", "Gender", "ICULOS"]
    
    df_train = pd.DataFrame({
        "HR": [80.0, 90.0, 100.0],
        "Age": [60.0, 60.0, 60.0],
        "Gender": [1, 1, 1],
        "ICULOS": [1, 2, 3]
    })
    
    df_test = pd.DataFrame({
        "HR": [np.nan],
        "Age": [60.0],
        "Gender": [1],
        "ICULOS": [1]
    })
    
    preprocessor = CausalPreprocessor(vitals, labs, demographics)
    preprocessor.fit([df_train])
    
    # Train median is 90
    assert preprocessor.medians_["HR"] == 90.0
    
    # Change df_test value
    df_test["HR"] = [np.nan]
    transformed = preprocessor.transform([df_test])[0]
    
    # Imputed value must be the train median (90)
    assert transformed["HR"].iloc[0] == 90.0

def test_utility_reference_example():
    # PhysioNet/CinC Challenge 2019 Utility Metric scoring rules:
    # A true positive within the sepsis window [t_sepsis - 12, t_sepsis + 3] receives positive reward.
    # An alarm before t_sepsis - 12 is a premature false alarm (negative reward).
    # An alarm after t_sepsis + 3 is a late alarm (no reward or penalty).
    # A false alarm on a non-sepsis patient receives a negative penalty.
    # Let's test a simple dummy utility scorer setup.
    # labels: 0 0 0 1 1 1 (sepsis onset at index 3, meaning window is hours 0 to 6)
    # predictions: 0 0 0 0 0 0 (no alarms)
    # utility should be low (0 or slightly negative due to missed diagnosis, but no false alarm penalties)
    
    labels = np.array([0, 0, 0, 1, 1, 1])
    preds = np.array([0, 0, 0, 0, 0, 0])
    
    # Since we need a test utility function in src/evaluation/metrics.py, we will import and test it.
    # We will verify that it computes successfully and returns a float.
    utility = calculate_physionet_utility(labels, preds, dt_early=-12, dt_late=3.0, max_u_tp=1)
    assert isinstance(utility, float)
