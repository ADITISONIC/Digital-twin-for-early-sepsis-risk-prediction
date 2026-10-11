import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)

class CausalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, vitals, labs, demographics, max_forward_fill_gap=24):
        self.vitals = vitals
        self.labs = labs
        self.demographics = demographics
        self.max_forward_fill_gap = max_forward_fill_gap
        self.cols_to_impute = vitals + labs
        self.medians_ = {}
        self.demographics_medians_ = {}
        
    def fit(self, X_list, y=None):
        """
        X_list: list of pandas DataFrames, one per patient.
        We compute the median value of each clinical variable across all patient-hours in X_list.
        """
        logger.info("Fitting CausalPreprocessor medians...")
        
        # Concatenate all patient-hours to compute global training medians
        concatenated = pd.concat([df[self.cols_to_impute] for df in X_list], axis=0, ignore_index=True)
        for col in self.cols_to_impute:
            self.medians_[col] = concatenated[col].median()
            # If a column is completely empty, default to 0.0
            if pd.isna(self.medians_[col]):
                self.medians_[col] = 0.0
        
        # Demographics fallbacks (like Unit1, Unit2, etc.)
        concatenated_demo = pd.concat([df[self.demographics] for df in X_list], axis=0, ignore_index=True)
        for col in self.demographics:
            self.demographics_medians_[col] = concatenated_demo[col].median()
            if pd.isna(self.demographics_medians_[col]):
                self.demographics_medians_[col] = 0.0
                
        logger.info("Medians fitted successfully.")
        return self
        
    def transform(self, X_list):
        """
        X_list: list of pandas DataFrames.
        Transforms each patient DataFrame causally and in a highly vectorized grouped way.
        """
        logger.info("Transforming cohort of %d patients using vectorized pipeline...", len(X_list))
        
        # Ensure copy of input and add dummy PatientID/HospitalID if missing
        X_list_proc = []
        for idx, df in enumerate(X_list):
            df_copy = df.copy()
            if "PatientID" not in df_copy.columns:
                df_copy["PatientID"] = f"dummy_{idx}"
            if "HospitalID" not in df_copy.columns:
                df_copy["HospitalID"] = "dummy"
            X_list_proc.append(df_copy)
            
        # Keep track of original patient order to return list in the same order
        patient_order = [df["PatientID"].iloc[0] for df in X_list_proc]
        
        # Concatenate all patient dataframes
        df_all = pd.concat(X_list_proc, axis=0, ignore_index=True)
        df_all = df_all.sort_values(["PatientID", "ICULOS"])
        
        # Dictionary to accumulate new columns and prevent fragmentation warnings
        new_cols = {}
        
        # Process vitals and labs
        for col in self.cols_to_impute:
            is_observed = df_all[col].notna()
            
            # Temporary column to hold observed hour
            df_all["_temp_obs_hour"] = np.where(is_observed, df_all["ICULOS"], np.nan)
            
            # Grouped ffill to find the hour of last observation
            last_obs_hour = df_all.groupby("PatientID", sort=False)["_temp_obs_hour"].ffill()
            
            # recency = ICULOS - last_obs_hour
            recency = df_all["ICULOS"] - last_obs_hour
            
            # Fallback for NaNs (before first observation occurs)
            fallback = 100.0 + df_all["ICULOS"]
            recency = np.where(recency.isna(), fallback, recency)
            
            new_cols[f"{col}_recency"] = recency.astype(float)
            new_cols[f"{col}_missing"] = (1.0 - is_observed.astype(float)).values
            
            # Causal ffill of clinical variables grouped by patient
            df_all[col] = df_all.groupby("PatientID", sort=False)[col].ffill(limit=self.max_forward_fill_gap)
            
            # Median fallback
            df_all[col] = df_all[col].fillna(self.medians_[col])
            
        # Clean up temporary column
        if "_temp_obs_hour" in df_all.columns:
            df_all = df_all.drop(columns=["_temp_obs_hour"])
            
        # Demographics ffill & imputation (e.g. Unit1, Unit2 might be NaN sometimes)
        for col in self.demographics:
            df_all[col] = df_all.groupby("PatientID", sort=False)[col].ffill()
            df_all[col] = df_all[col].fillna(self.demographics_medians_[col])
            
        # Concatenate all new columns at once to prevent fragmentation
        if new_cols:
            new_cols_df = pd.DataFrame(new_cols, index=df_all.index)
            df_all = pd.concat([df_all, new_cols_df], axis=1)
            
        # Split back to list of dataframes in the original order
        groups = {name: group for name, group in df_all.groupby("PatientID", sort=False)}
        transformed_list = [groups[p_id] for p_id in patient_order]
        
        logger.info("Transformation complete.")
        return transformed_list
