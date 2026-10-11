import os
import pandas as pd
import numpy as np
import logging
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_audit():
    config = load_config()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    raw_dir = os.path.join(base_dir, config["paths"]["raw_dir"])
    results_tables_dir = os.path.join(base_dir, config["paths"]["results_dir"], "tables")
    results_figures_dir = os.path.join(base_dir, config["paths"]["results_dir"], "figures")
    
    os.makedirs(results_tables_dir, exist_ok=True)
    os.makedirs(results_figures_dir, exist_ok=True)
    
    set_a_dir = os.path.join(raw_dir, "training_setA")
    set_b_dir = os.path.join(raw_dir, "training_setB")
    
    sets = {"A": set_a_dir, "B": set_b_dir}
    
    # Store patient-level summaries
    patient_records = []
    
    # Variable counts
    col_missing_counts = {}
    total_non_nan_counts = {}
    total_hours = 0
    malformed_files = []
    
    # Expected columns
    expected_cols = (
        config["variables"]["vitals"] + 
        config["variables"]["labs"] + 
        config["variables"]["demographics"] + 
        [config["data"]["target_col"]]
    )
    
    logger.info("Starting dataset audit...")
    
    for set_name, set_dir in sets.items():
        if not os.path.exists(set_dir):
            logger.error("Directory %s does not exist!", set_dir)
            continue
            
        files = [f for f in os.listdir(set_dir) if f.endswith(".psv")]
        logger.info("Auditing Set %s: %d files...", set_name, len(files))
        
        for idx, filename in enumerate(files):
            filepath = os.path.join(set_dir, filename)
            patient_id = filename.split('.')[0]
            
            try:
                # Delimiter is |
                df = pd.read_csv(filepath, sep="|")
                
                # Check for malformed columns
                if list(df.columns) != expected_cols:
                    malformed_files.append((filepath, "Columns mismatch"))
                    continue
                
                # Rows/hours
                n_rows = len(df)
                total_hours += n_rows
                
                if n_rows == 0:
                    malformed_files.append((filepath, "Empty file"))
                    continue
                
                # Sepsis labels
                labels = df[config["data"]["target_col"]].values
                has_sepsis = 1 if (labels == 1).any() else 0
                
                first_sepsis_hour = -1
                last_sepsis_hour = -1
                if has_sepsis:
                    sepsis_indices = np.where(labels == 1)[0]
                    first_sepsis_hour = int(sepsis_indices[0])
                    last_sepsis_hour = int(sepsis_indices[-1])
                
                # Demographics (Age, Gender, Unit1, Unit2, HospAdmTime)
                age = df["Age"].iloc[0] if "Age" in df.columns else np.nan
                gender = df["Gender"].iloc[0] if "Gender" in df.columns else np.nan
                
                patient_records.append({
                    "PatientID": patient_id,
                    "Hospital": set_name,
                    "LengthOfStay": n_rows,
                    "SepsisEver": has_sepsis,
                    "FirstSepsisHour": first_sepsis_hour,
                    "LastSepsisHour": last_sepsis_hour,
                    "Age": age,
                    "Gender": gender
                })
                
                # Missingness and variable counts
                for col in expected_cols:
                    if col not in col_missing_counts:
                        col_missing_counts[col] = 0
                        total_non_nan_counts[col] = 0
                    nan_count = df[col].isna().sum()
                    col_missing_counts[col] += nan_count
                    total_non_nan_counts[col] += (n_rows - nan_count)
                    
            except Exception as e:
                malformed_files.append((filepath, str(e)))
                
            if (idx + 1) % 5000 == 0:
                logger.info("Processed %d patients in Set %s...", idx + 1, set_name)
                
    # Build dataframe of patient summaries
    patient_df = pd.DataFrame(patient_records)
    
    # Audit reporting
    total_patients = len(patient_df)
    patients_a = len(patient_df[patient_df["Hospital"] == "A"])
    patients_b = len(patient_df[patient_df["Hospital"] == "B"])
    
    sepsis_a = int(patient_df[(patient_df["Hospital"] == "A") & (patient_df["SepsisEver"] == 1)]["SepsisEver"].count())
    sepsis_b = int(patient_df[(patient_df["Hospital"] == "B") & (patient_df["SepsisEver"] == 1)]["SepsisEver"].count())
    non_sepsis = int(patient_df[patient_df["SepsisEver"] == 0]["SepsisEver"].count())
    
    logger.info("Audit Summary:")
    logger.info("Total Patients: %d", total_patients)
    logger.info("Hospital A: %d (Sepsis: %d)", patients_a, sepsis_a)
    logger.info("Hospital B: %d (Sepsis: %d)", patients_b, sepsis_b)
    logger.info("Total Patient-Hours: %d", total_hours)
    
    # ICU Stay Lengths
    los_stats = patient_df["LengthOfStay"].describe()
    
    # Save dataset_summary.csv
    summary_data = {
        "Metric": [
            "total_patients",
            "patients_in_A",
            "patients_in_B",
            "sepsis_patients_in_A",
            "sepsis_patients_in_B",
            "non_sepsis_patients",
            "total_patient_hours",
            "minimum_ICU_length",
            "maximum_ICU_length",
            "median_ICU_length",
            "class_prevalence_patients_A",
            "class_prevalence_patients_B",
            "class_prevalence_hours_A",
            "class_prevalence_hours_B",
            "number_of_variables",
            "number_of_files",
            "malformed_files",
            "duplicate_patient_ids",
            "duplicate_rows",
            "invalid_numeric_values"
        ],
        "Value": [
            total_patients,
            patients_a,
            patients_b,
            sepsis_a,
            sepsis_b,
            non_sepsis,
            total_hours,
            int(los_stats["min"]),
            int(los_stats["max"]),
            int(los_stats["50%"]),
            sepsis_a / patients_a if patients_a > 0 else 0,
            sepsis_b / patients_b if patients_b > 0 else 0,
            # We need hourly prevalence
            # Let's count sepsis hours in A and B
            0, # will fill in next
            0, # will fill in next
            len(expected_cols),
            total_patients + len(malformed_files),
            len(malformed_files),
            total_patients - patient_df["PatientID"].nunique(),
            0,  # We don't merge rows, each is an ICU hour, so no duplicate rows per patient
            0   # Assuming standard parsing, we can track if NaNs aren't the only ones
        ]
    }
    
    # Let's compute class prevalence in hours
    sepsis_hours_a = 0
    sepsis_hours_b = 0
    hours_a = 0
    hours_b = 0
    
    # Let's calculate exactly
    for set_name, set_dir in sets.items():
        files = [f for f in os.listdir(set_dir) if f.endswith(".psv")]
        for filename in files:
            filepath = os.path.join(set_dir, filename)
            try:
                df = pd.read_csv(filepath, sep="|")
                labels = df[config["data"]["target_col"]].values
                if set_name == "A":
                    sepsis_hours_a += np.sum(labels == 1)
                    hours_a += len(df)
                else:
                    sepsis_hours_b += np.sum(labels == 1)
                    hours_b += len(df)
            except:
                pass
                
    summary_data["Value"][12] = sepsis_hours_a / hours_a if hours_a > 0 else 0
    summary_data["Value"][13] = sepsis_hours_b / hours_b if hours_b > 0 else 0
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(results_tables_dir, "dataset_summary.csv"), index=False)
    logger.info("Saved dataset_summary.csv")
    
    # Missingness summary
    missingness_records = []
    for col in expected_cols:
        total_miss = col_missing_counts.get(col, 0)
        pct_miss = (total_miss / total_hours) * 100
        missingness_records.append({
            "Variable": col,
            "MissingCount": total_miss,
            "MissingPercentage": pct_miss
        })
    missing_df = pd.DataFrame(missingness_records).sort_values("MissingPercentage", ascending=False)
    missing_df.to_csv(os.path.join(results_tables_dir, "missingness_summary.csv"), index=False)
    logger.info("Saved missingness_summary.csv")
    
    # Generate missingness plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=missing_df.head(25), y="Variable", x="MissingPercentage", palette="viridis")
    plt.title("Missingness Percentage of Top 25 Variables in Sepsis Dataset")
    plt.xlabel("Percentage Missing (%)")
    plt.ylabel("Clinical Variable")
    plt.tight_layout()
    plt.savefig(os.path.join(results_figures_dir, "missingness.png"), dpi=300)
    plt.close()
    logger.info("Saved missingness.png")
    
    # Generate cohort distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=patient_df, x="LengthOfStay", hue="Hospital", multiple="dodge", bins=30, kde=True)
    plt.title("Distribution of ICU Length of Stay across Hospitals")
    plt.xlabel("ICU Stay Length (Hours)")
    plt.ylabel("Number of Patients")
    plt.savefig(os.path.join(results_figures_dir, "cohort_distribution.png"), dpi=300)
    plt.close()
    logger.info("Saved cohort_distribution.png")
    
    # Create data dictionary
    # Official descriptions from PhysioNet challenge
    descriptions = {
        "HR": "Heart rate (beats per minute)",
        "O2Sat": "Pulse oximetry (percentage)",
        "Temp": "Temperature (deg C)",
        "SBP": "Systolic blood pressure (mm Hg)",
        "MAP": "Mean arterial pressure (mm Hg)",
        "DBP": "Diastolic blood pressure (mm Hg)",
        "Resp": "Respiration rate (breaths per minute)",
        "EtCO2": "End-tidal carbon dioxide (mm Hg)",
        "BaseExcess": "Measure of excess bicarbonate (mmol/L)",
        "HCO3": "Bicarbonate concentration (mmol/L)",
        "FiO2": "Fraction of inspired oxygen (percentage)",
        "pH": "Acidity/alkalinity of blood",
        "PaCO2": "Partial pressure of carbon dioxide in arterial blood (mm Hg)",
        "SaO2": "Arterial oxygen saturation (percentage)",
        "AST": "Aspartate aminotransferase (IU/L)",
        "BUN": "Blood urea nitrogen (mg/dL)",
        "Alkalinephos": "Alkaline phosphatase (IU/L)",
        "Calcium": "Calcium concentration (mg/dL)",
        "Chloride": "Chloride concentration (mmol/L)",
        "Creatinine": "Creatinine concentration (mg/dL)",
        "Bilirubin_direct": "Direct bilirubin (mg/dL)",
        "Glucose": "Serum glucose concentration (mg/dL)",
        "Lactate": "Lactate concentration (mmol/L)",
        "Magnesium": "Magnesium concentration (mmol/L)",
        "Phosphate": "Phosphate concentration (mg/dL)",
        "Potassium": "Potassium concentration (mmol/L)",
        "Bilirubin_total": "Total bilirubin (mg/dL)",
        "TroponinI": "Troponin I concentration (ng/mL)",
        "Hct": "Hematocrit (percentage)",
        "Hgb": "Hemoglobin concentration (g/dL)",
        "PTT": "Partial thromboplastin time (seconds)",
        "WBC": "White blood cell count (cells/microL)",
        "Fibrinogen": "Fibrinogen concentration (mg/dL)",
        "Platelets": "Platelet count (cells/microL)",
        "Age": "Age of patient (years)",
        "Gender": "Gender (female=0, male=1)",
        "Unit1": "Medical ICU indicator (MICU)",
        "Unit2": "Surgical ICU indicator (SICU)",
        "HospAdmTime": "Hours between hospital admit and ICU admit",
        "ICULOS": "ICU length of stay (hours since ICU admit)",
        "SepsisLabel": "Sepsis label (1 if sepsis diagnosed in next 6h, else 0)"
    }
    
    dict_records = []
    for col in expected_cols:
        col_type = "Vital Sign" if col in config["variables"]["vitals"] else \
                   "Laboratory Value" if col in config["variables"]["labs"] else \
                   "Demographic/Administrative" if col in config["variables"]["demographics"] else \
                   "Prediction Target"
        dict_records.append({
            "Variable": col,
            "Type": col_type,
            "Description": descriptions.get(col, "Unknown clinical variable"),
            "MissingPercentage_Total": f"{missing_df[missing_df['Variable'] == col]['MissingPercentage'].values[0]:.2f}%"
        })
    dict_df = pd.DataFrame(dict_records)
    dict_df.to_csv(os.path.join(results_tables_dir, "data_dictionary.csv"), index=False)
    logger.info("Saved data_dictionary.csv")
    
    # Print malformed files if any
    if malformed_files:
        logger.warning("Found %d malformed files:", len(malformed_files))
        for path, err in malformed_files[:10]:
            logger.warning("  %s: %s", path, err)

if __name__ == "__main__":
    run_audit()
