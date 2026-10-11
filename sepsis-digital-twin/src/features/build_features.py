import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def build_dynamic_features_pipeline(df_list, vitals, labs, target_col="SepsisLabel", patient_id_col="PatientID", hospital_id_col="HospitalID"):
    """
    df_list: List of preprocessed patient DataFrames.
    Runs feature engineering at the cohort level in a fully vectorized way.
    """
    logger.info("Building dynamic features using vectorized cohort pipeline...")
    
    # Check if PatientID and HospitalID are present. If not, add them (for unit tests)
    df_list_proc = []
    for idx, df in enumerate(df_list):
        df_copy = df.copy()
        if patient_id_col not in df_copy.columns:
            df_copy[patient_id_col] = f"dummy_{idx}"
        if hospital_id_col not in df_copy.columns:
            df_copy[hospital_id_col] = "dummy"
        df_list_proc.append(df_copy)
        
    # Original patient order to split back
    patient_order = [df[patient_id_col].iloc[0] for df in df_list_proc]
    
    # Concatenate cohort
    df_all = pd.concat(df_list_proc, axis=0, ignore_index=True)
    df_all = df_all.sort_values([patient_id_col, "ICULOS"])
    
    continuous_cols = vitals + labs
    new_features = {}
    
    # We will group by PatientID for shift and cumsum operations
    patient_groupby = df_all.groupby(patient_id_col, sort=False)
    
    # ICU length of stay is the time index
    iculos = df_all["ICULOS"].values
    
    for col in continuous_cols:
        x = df_all[col].values
        
        # First value per patient (grouped transform)
        first_val = patient_groupby[col].transform("first")
        
        # Cumulative sums grouped by patient
        cumsum_x = patient_groupby[col].cumsum()
        cumsum_x2 = (df_all[col]**2).groupby(df_all[patient_id_col], sort=False).cumsum()
        
        # Rolling mean & std helper
        for w in [3, 6]:
            # Shifted cumsums grouped by patient
            shift_cum_x = cumsum_x.groupby(df_all[patient_id_col], sort=False).shift(w).fillna(0.0)
            shift_cum_x2 = cumsum_x2.groupby(df_all[patient_id_col], sort=False).shift(w).fillna(0.0)
            
            sum_x = (cumsum_x - shift_cum_x).values
            sum_x2 = (cumsum_x2 - shift_cum_x2).values
            
            n = np.minimum(w, iculos)
            
            # Mean
            mean_val = sum_x / n
            new_features[f"{col}_mean_{w}h"] = mean_val
            
            # Std (sample variance formula: (sum_x2 - (sum_x^2)/n)/(n-1))
            variance = (sum_x2 - (sum_x**2) / n) / np.maximum(1.0, n - 1)
            std_val = np.sqrt(np.maximum(0.0, variance))
            std_val = np.where(n > 1, std_val, 0.0)
            new_features[f"{col}_std_{w}h"] = std_val
            
        # Shifts grouped by patient for trend and change
        y0 = df_all[col]
        y1 = patient_groupby[col].shift(1).fillna(first_val)
        y2 = patient_groupby[col].shift(2).fillna(first_val)
        y3 = patient_groupby[col].shift(3).fillna(first_val)
        y4 = patient_groupby[col].shift(4).fillna(first_val)
        y5 = patient_groupby[col].shift(5).fillna(first_val)
        
        # Current
        new_features[f"{col}_current"] = x
        
        # Delta
        new_features[f"{col}_delta_3h"] = (y0 - y3).values
        new_features[f"{col}_delta_6h"] = (y0 - y5).values
        
        # Slope
        new_features[f"{col}_slope_3h"] = ((y0 - y2) / 2.0).values
        new_features[f"{col}_slope_6h"] = ((5.0 * (y0 - y5) + 3.0 * (y1 - y4) + (y2 - y3)) / 35.0).values
        
    # Concatenate new features
    new_features_df = pd.DataFrame(new_features, index=df_all.index)
    df_all = pd.concat([df_all, new_features_df], axis=1)
    
    # Split back into patient list in original order
    groups = {name: group for name, group in df_all.groupby(patient_id_col, sort=False)}
    transformed_list = [groups[p_id] for p_id in patient_order]
    
    logger.info("Feature engineering complete.")
    return transformed_list

def build_dynamic_features_for_patient(df, vitals, labs, target_col="SepsisLabel", patient_id_col="PatientID", hospital_id_col="HospitalID"):
    """
    df: Preprocessed DataFrame for a single patient.
    """
    df_copy = df.copy()
    if patient_id_col not in df_copy.columns:
        df_copy[patient_id_col] = "dummy"
    if hospital_id_col not in df_copy.columns:
        df_copy[hospital_id_col] = "dummy"
    return build_dynamic_features_pipeline(
        [df_copy], vitals, labs, target_col, patient_id_col, hospital_id_col
    )[0]
