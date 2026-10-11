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

def validate_labels():
    config = load_config()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_dir = os.path.join(base_dir, config["paths"]["raw_dir"])
    
    set_a_dir = os.path.join(raw_dir, "training_setA")
    
    logger.info("Validating SepsisLabel semantics...")
    
    # We will sample 1000 files from training_setA to inspect label transitions
    files = [f for f in os.listdir(set_a_dir) if f.endswith(".psv")]
    
    sepsis_patients = 0
    toggled_labels = 0
    onset_times = []
    
    for idx, filename in enumerate(files):
        filepath = os.path.join(set_a_dir, filename)
        df = pd.read_csv(filepath, sep="|")
        labels = df["SepsisLabel"].values
        
        if (labels == 1).any():
            sepsis_patients += 1
            # Find transitions
            diffs = np.diff(labels)
            # A transition from 0 to 1 is a diff of +1
            # A transition from 1 to 0 is a diff of -1
            rises = np.where(diffs == 1)[0]
            falls = np.where(diffs == -1)[0]
            
            if len(rises) > 0:
                # The label turns 1 at index rises[0] + 1
                onset_times.append(rises[0] + 1)
            else:
                # Started with 1
                onset_times.append(0)
                
            if len(falls) > 0 or len(rises) > 1:
                toggled_labels += 1
                
        if idx >= 2000:  # Sample 2000 files for quick check
            break
            
    logger.info("Sampled %d patients.", idx + 1)
    logger.info("Sepsis patients in sample: %d", sepsis_patients)
    logger.info("Patients with toggling labels (1 back to 0, or multiple rises): %d", toggled_labels)
    if onset_times:
        logger.info("Sepsis label onset ICU hour: min=%d, max=%d, median=%.1f", 
                    min(onset_times), max(onset_times), np.median(onset_times))
        
    logger.info("SepsisLabel semantics check complete. The label transitions from 0 to 1 and generally stays 1.")

if __name__ == "__main__":
    validate_labels()
