import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


class D1DNNDetector:

    # -------------------------------------------------
    # D1 feature order
    # -------------------------------------------------

    FEATURES = [
        "switch",
        "pktcount",
        "bytecount",
        "dur",
        "dur_nsec",
        "tot_dur",
        "flows",
        "packetins",
        "pktperflow",
        "byteperflow",
        "pktrate",
        "Pairflow",
        "Protocol",
        "port_no",
        "tx_bytes",
        "rx_bytes",
        "tx_kbps",
        "rx_kbps",
        "tot_kbps"
    ]

    # -------------------------------------------------
    # Protocol encoding
    # Same encoding used during D1 training
    # -------------------------------------------------

    PROTOCOL_MAPPING = {
        "ICMP": 0,
        "TCP": 1,
        "UDP": 2
    }

    def __init__(self):

        # -------------------------------------------------
        # Project root
        # -------------------------------------------------

        project_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                ".."
            )
        )

        # -------------------------------------------------
        # Model and preprocessing artifacts
        # -------------------------------------------------

        self.model_path = os.path.join(
            project_root,
            "results",
            "models",
            "dnn",
            "dnn_dataset1_best.keras"
        )

        self.medians_path = os.path.join(
            project_root,
            "results",
            "preprocessing",
            "dataset1_medians.pkl"
        )

        self.scaler_path = os.path.join(
            project_root,
            "results",
            "preprocessing",
            "dataset1_scaler.pkl"
        )

        # -------------------------------------------------
        # Load trained DNN model
        # -------------------------------------------------

        self.model = tf.keras.models.load_model(
            self.model_path
        )

        # -------------------------------------------------
        # Load preprocessing artifacts
        # -------------------------------------------------

        self.medians = joblib.load(
            self.medians_path
        )

        self.scaler = joblib.load(
            self.scaler_path
        )

        print("D1 DNN detector loaded successfully.")

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    def predict(self, record):

        # ---------------------------------------------
        # Build feature dataframe
        # ---------------------------------------------

        data = {}

        for feature in self.FEATURES:

            value = record.get(
                feature,
                np.nan
            )

            data[feature] = value

        df = pd.DataFrame(
            [data],
            columns=self.FEATURES
        )

        # ---------------------------------------------
        # Convert Protocol to training representation
        #
        # ICMP -> 0
        # TCP  -> 1
        # UDP  -> 2
        #
        # This is required because the raw D1 dataset
        # stores Protocol as text.
        # ---------------------------------------------

        if "Protocol" in df.columns:

            if df["Protocol"].dtype == object:

                df["Protocol"] = (
                    df["Protocol"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .map(self.PROTOCOL_MAPPING)
                )

        # ---------------------------------------------
        # Convert all features to numeric
        # ---------------------------------------------

        for feature in self.FEATURES:

            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce"
            )

        # ---------------------------------------------
        # Replace infinite values
        # ---------------------------------------------

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # ---------------------------------------------
        # Median imputation
        # Same approach used during D1 preprocessing
        # ---------------------------------------------

        for feature in self.FEATURES:

            if pd.isna(df.at[0, feature]):

                df.at[0, feature] = self.medians[
                    feature
                ]

        # ---------------------------------------------
        # Scale using training scaler
        # ---------------------------------------------

        X = self.scaler.transform(
            df[self.FEATURES]
        )

        # ---------------------------------------------
        # DNN prediction
        # ---------------------------------------------

        probabilities = self.model.predict(
            X,
            verbose=0
        )[0]

        predicted_class = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_class]
        )

        # ---------------------------------------------
        # D1 labels
        #
        # 0 = NORMAL
        # 1 = MALICIOUS
        # ---------------------------------------------

        if predicted_class == 1:
            label = "MALICIOUS"
        else:
            label = "NORMAL"

        # ---------------------------------------------
        # Return prediction result
        # ---------------------------------------------

        return {
            "class": predicted_class,
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities.tolist()
        }