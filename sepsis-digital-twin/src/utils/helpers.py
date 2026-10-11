import os
import random
import numpy as np
import logging
import yaml

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config():
    # Resolved relative to this file
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # If using torch or tensorflow, set their seeds here too
    # torch.manual_seed(seed)
    # tf.random.set_seed(seed)
    
def get_patient_filepaths(config, patient_ids, hospital_id):
    """
    Returns absolute file paths for a list of patient IDs.
    hospital_id is either 'A' or 'B'.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_dir = os.path.join(base_dir, config["paths"]["raw_dir"])
    
    folder_name = "training_setA" if hospital_id == 'A' else "training_setB"
    folder_path = os.path.join(raw_dir, folder_name)
    
    paths = []
    for p_id in patient_ids:
        paths.append(os.path.join(folder_path, f"{p_id}.psv"))
    return paths
