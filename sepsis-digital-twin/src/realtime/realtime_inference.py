import os
import pickle
import numpy as np
import pandas as pd

from src.utils.helpers import load_config
from src.features.build_features import build_dynamic_features_pipeline


class RealtimeInference:

    def __init__(self):

        self.config = load_config()

        base_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                ".."
            )
        )

        models_dir = os.path.join(
            base_dir,
            self.config["paths"]["models_dir"]
        )

        model_path = os.path.join(
            models_dir,
            "xgboost_model.pkl"
        )

        print("Loading model:", model_path)

        with open(model_path, "rb") as f:
            artifact = pickle.load(f)

        self.model = artifact["model"]
        self.preprocessor = artifact["preprocessor"]
        self.feature_cols = artifact["feature_cols"]
        self.threshold = artifact["best_threshold"]
        self.scaler = artifact.get("scaler", None)

    def predict(self, patient_history):

        df = pd.DataFrame(patient_history)

        # ------------------------------------------------
        # Target column required by feature pipeline
        # ------------------------------------------------

        target_col = self.config["data"]["target_col"]

        if target_col not in df.columns:
            df[target_col] = 0

        # ------------------------------------------------
        # Preprocessing
        # ------------------------------------------------

        transformed = self.preprocessor.transform([df])

        # ------------------------------------------------
        # Dynamic feature engineering
        # ------------------------------------------------

        feature_dfs = build_dynamic_features_pipeline(
            transformed,
            vitals=self.config["variables"]["vitals"],
            labs=self.config["variables"]["labs"]
        )

        processed_df = feature_dfs[0]

        # ------------------------------------------------
        # Select trained features
        # ------------------------------------------------

        missing = [
            col
            for col in self.feature_cols
            if col not in processed_df.columns
        ]

        if missing:

            raise ValueError(
                "Missing model features: "
                + ", ".join(missing)
            )

        X = processed_df[self.feature_cols].values

        # ------------------------------------------------
        # Scaling
        # ------------------------------------------------

        if self.scaler is not None:

            X = self.scaler.transform(X)

        # ------------------------------------------------
        # Prediction
        # ------------------------------------------------

        probabilities = self.model.predict_proba(X)[:, 1]

        processed_df["PredictedRisk"] = probabilities

        processed_df["SepsisAlert"] = (
            probabilities >= self.threshold
        ).astype(int)

        return processed_df, probabilities