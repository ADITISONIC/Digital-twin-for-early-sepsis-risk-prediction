#!/bin/bash
set -e
export PYTHONPATH=.

# Sepsis early-warning project master execution script
echo "=========================================================="
echo "Starting Sepsis Early-Warning Research Pipeline Execution"
echo "=========================================================="

echo "Step 1: Generating Patient Cohort Splits..."
python3 src/data/build_splits.py

echo "Step 2: Training baseline Logistic Regression model..."
python3 src/models/train_lr.py

echo "Step 3: Training and tuning primary XGBoost model..."
python3 src/models/train_xgb.py

echo "Step 4: Running final evaluation on held-out Hospital B..."
python3 src/evaluation/evaluate_holdout.py

echo "Step 5: Running SHAP explainability and generating case studies..."
python3 src/explainability/shap_explain.py

echo "=========================================================="
echo "Sepsis Early-Warning Pipeline Execution Completed Successfully"
echo "=========================================================="
