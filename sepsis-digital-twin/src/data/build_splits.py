import os
import pandas as pd
import numpy as np
import logging
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_splits():
    config = load_config()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    raw_dir = os.path.join(base_dir, config["paths"]["raw_dir"])
    splits_dir = os.path.join(base_dir, config["paths"]["splits_dir"])
    os.makedirs(splits_dir, exist_ok=True)
    
    set_a_dir = os.path.join(raw_dir, "training_setA")
    set_b_dir = os.path.join(raw_dir, "training_setB")
    
    # Get all patient IDs
    patients_a = sorted([f.split('.')[0] for f in os.listdir(set_a_dir) if f.endswith(".psv")])
    patients_b = sorted([f.split('.')[0] for f in os.listdir(set_b_dir) if f.endswith(".psv")])
    
    logger.info("Total Hospital A patients: %d", len(patients_a))
    logger.info("Total Hospital B patients: %d", len(patients_b))
    
    # Shuffle Hospital A patients deterministically
    np.random.seed(config["random_seed"])
    shuffled_a = np.random.permutation(patients_a)
    
    # 80/20 train/validation split
    split_idx = int(len(shuffled_a) * 0.8)
    train_patients = sorted(shuffled_a[:split_idx])
    val_patients = sorted(shuffled_a[split_idx:])
    
    logger.info("Hospital A splits: Train=%d, Validation=%d", len(train_patients), len(val_patients))
    
    # Verify no overlap
    set_train = set(train_patients)
    set_val = set(val_patients)
    set_holdout = set(patients_b)
    
    assert len(set_train.intersection(set_val)) == 0, "Overlap between train and validation!"
    assert len(set_train.intersection(set_holdout)) == 0, "Overlap between train and holdout!"
    assert len(set_val.intersection(set_holdout)) == 0, "Overlap between validation and holdout!"
    
    # Save splits
    pd.DataFrame({"PatientID": train_patients}).to_csv(os.path.join(splits_dir, "train_patients.csv"), index=False)
    pd.DataFrame({"PatientID": val_patients}).to_csv(os.path.join(splits_dir, "validation_patients.csv"), index=False)
    pd.DataFrame({"PatientID": patients_b}).to_csv(os.path.join(splits_dir, "holdout_hospitalB_patients.csv"), index=False)
    
    logger.info("Saved patient splits to %s", splits_dir)

if __name__ == "__main__":
    build_splits()
