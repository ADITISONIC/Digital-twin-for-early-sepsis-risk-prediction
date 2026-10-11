import os
import pandas as pd
import numpy as np
import logging
from src.utils.helpers import load_config

logger = logging.getLogger(__name__)

def load_patient_file(filepath, patient_id, hospital_id, expected_cols=None):
    """
    Loads a single patient's PSV file.
    Validates columns and adds PatientID and HospitalID columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Patient file not found: {filepath}")
        
    try:
        df = pd.read_csv(filepath, sep="|")
        
        # Check columns
        if expected_cols is not None:
            if list(df.columns) != expected_cols:
                raise ValueError(f"Columns mismatch in {filepath}. Expected {expected_cols}, got {list(df.columns)}")
                
        # Add identifiers
        df["PatientID"] = patient_id
        df["HospitalID"] = hospital_id
        
        # Chronological hour ordering is preserved by row order (ICULOS column is monotonic increasing)
        df = df.sort_values("ICULOS")
        return df
        
    except Exception as e:
        logger.error("Failed to load file %s: %s", filepath, str(e))
        raise

def load_cohort_data(patient_ids, hospital_id, config=None, limit=None):
    """
    Loads all patient files for a cohort as a list of DataFrames.
    If limit is provided, loads only the first 'limit' patients.
    """
    if config is None:
        config = load_config()
        
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_dir = os.path.join(base_dir, config["paths"]["raw_dir"])
    folder_name = "training_setA" if hospital_id == 'A' else "training_setB"
    folder_path = os.path.join(raw_dir, folder_name)
    
    expected_cols = (
        config["variables"]["vitals"] + 
        config["variables"]["labs"] + 
        config["variables"]["demographics"] + 
        [config["data"]["target_col"]]
    )
    
    df_list = []
    
    if limit is not None:
        patient_ids = patient_ids[:limit]
        
    logger.info("Loading cohort data for %d patients from Hospital %s...", len(patient_ids), hospital_id)
    
    for idx, p_id in enumerate(patient_ids):
        filepath = os.path.join(folder_path, f"{p_id}.psv")
        try:
            df = load_patient_file(filepath, p_id, hospital_id, expected_cols)
            df_list.append(df)
        except Exception as e:
            logger.error("Error loading patient %s: %s", p_id, str(e))
            raise
            
        if (idx + 1) % 5000 == 0:
            logger.info("Loaded %d / %d patients...", idx + 1, len(patient_ids))
            
    logger.info("Loaded %d patients successfully.", len(df_list))
    return df_list
