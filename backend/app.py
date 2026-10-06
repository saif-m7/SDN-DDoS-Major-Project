import os

import pandas as pd

from flask import Flask, jsonify
from flask_cors import CORS





app = Flask(__name__)
CORS(app)





# ============================================================

# PATH CONFIGURATION

# ============================================================



BASE_DIR = os.path.abspath(

    os.path.join(os.path.dirname(__file__), "..")

)



ANALYTICS_FILE = os.path.join(

    BASE_DIR,

    "sdn",

    "results",

    "analytics",

    "traffic_bandwidth.csv"

)





# ============================================================

# HOME

# ============================================================



@app.route("/")

def home():



    return jsonify({

        "message": "SDN DDoS Detection Backend API is running",

        "status": "ACTIVE"

    })





# ============================================================

# API 1: DASHBOARD OVERVIEW

# ============================================================



@app.route("/api/dashboard/overview")

def dashboard_overview():



    if not os.path.exists(ANALYTICS_FILE):



        return jsonify({

            "error": "Analytics file not found",

            "path": ANALYTICS_FILE

        }), 404



    try:



        df = pd.read_csv(ANALYTICS_FILE)



        if df.empty:



            return jsonify({

                "network_status": "NO_DATA",

                "total_traffic": 0,

                "current_bandwidth": 0,

                "peak_bandwidth": 0,

                "normal_flows": 0,

                "malicious_flows": 0,

                "attacks_detected": 0,

                "attacks_mitigated": 0

            })



        # --------------------------------------------------

        # D1 detection records

        # --------------------------------------------------



        detection_df = df[

            df["d1_prediction"].notna()

        ].copy()



        normal_flows = int(

            (

                detection_df["d1_prediction"]

                .astype(str)

                .str.upper()

                == "NORMAL"

            ).sum()

        )



        malicious_flows = int(

            (

                detection_df["d1_prediction"]

                .astype(str)

                .str.upper()

                == "MALICIOUS"

            ).sum()

        )



        attacks_detected = malicious_flows



        attacks_mitigated = int(

            (

                detection_df["mitigation_status"]

                .astype(str)

                .str.upper()

                == "MITIGATED"

            ).sum()

        )



        # --------------------------------------------------

        # Traffic and bandwidth records

        # --------------------------------------------------



        traffic_df = df[

            df["total_kbps"].notna()

        ].copy()



        current_bandwidth = 0.0

        peak_bandwidth = 0.0

        total_traffic = 0.0



        if not traffic_df.empty:



            traffic_df["total_kbps"] = pd.to_numeric(

                traffic_df["total_kbps"],

                errors="coerce"

            )



            traffic_df = traffic_df[

                traffic_df["total_kbps"].notna()

            ].copy()



            traffic_df["timestamp"] = pd.to_datetime(

                traffic_df["timestamp"],

                errors="coerce"

            )



            traffic_df = traffic_df[

                traffic_df["timestamp"].notna()

            ].copy()



            if not traffic_df.empty:



                traffic_df = traffic_df.sort_values(

                    "timestamp"

                )



                total_traffic = float(

                    traffic_df["total_kbps"].sum()

                )



                peak_bandwidth = float(

                    traffic_df["total_kbps"].max()

                )



                non_zero_traffic = traffic_df[

                    traffic_df["total_kbps"] > 0

                ]



                if not non_zero_traffic.empty:



                    latest_row = non_zero_traffic.iloc[-1]



                    current_bandwidth = float(

                        latest_row["total_kbps"]

                    )



        # --------------------------------------------------

        # Response

        # --------------------------------------------------



        return jsonify({

            "network_status": "ACTIVE",

            "total_traffic": round(total_traffic, 2),

            "current_bandwidth": round(current_bandwidth, 2),

            "peak_bandwidth": round(peak_bandwidth, 2),

            "normal_flows": normal_flows,

            "malicious_flows": malicious_flows,

            "attacks_detected": attacks_detected,

            "attacks_mitigated": attacks_mitigated

        })



    except Exception as e:



        return jsonify({

            "error": str(e)

        }), 500





# ============================================================

# API 2: LIVE TRAFFIC

# ============================================================



@app.route("/api/traffic/live")

def traffic_live():



    if not os.path.exists(ANALYTICS_FILE):



        return jsonify({

            "error": "Analytics file not found",

            "path": ANALYTICS_FILE

        }), 404



    try:



        df = pd.read_csv(ANALYTICS_FILE)



        if df.empty:



            return jsonify({

                "status": "NO_DATA",

                "count": 0,

                "records": []

            })



        traffic_df = df[

            df["total_kbps"].notna()

        ].copy()



        if traffic_df.empty:



            return jsonify({

                "status": "NO_TRAFFIC_DATA",

                "count": 0,

                "records": []

            })



        # --------------------------------------------------

        # Convert numeric fields

        # --------------------------------------------------



        numeric_columns = [

            "switch",

            "port",

            "rx_bytes",

            "tx_bytes",

            "rx_packets",

            "tx_packets",

            "rx_kbps",

            "tx_kbps",

            "total_kbps"

        ]



        for column in numeric_columns:



            if column in traffic_df.columns:



                traffic_df[column] = pd.to_numeric(

                    traffic_df[column],

                    errors="coerce"

                )



        # --------------------------------------------------

        # Convert timestamp

        # --------------------------------------------------



        traffic_df["timestamp"] = pd.to_datetime(

            traffic_df["timestamp"],

            errors="coerce"

        )



        traffic_df = traffic_df[

            traffic_df["timestamp"].notna()

        ].copy()



        # --------------------------------------------------

        # Prefer non-zero traffic

        # --------------------------------------------------



        non_zero_traffic = traffic_df[

            traffic_df["total_kbps"] > 0

        ].copy()



        if not non_zero_traffic.empty:



            traffic_df = non_zero_traffic



        # --------------------------------------------------

        # Sort newest first

        # --------------------------------------------------



        traffic_df = traffic_df.sort_values(

            "timestamp",

            ascending=False

        )



        # --------------------------------------------------

        # Latest 20 records

        # --------------------------------------------------



        traffic_df = traffic_df.head(20).copy()



        traffic_df["timestamp"] = (

            traffic_df["timestamp"]

            .dt.strftime("%Y-%m-%d %H:%M:%S")

        )



        # --------------------------------------------------

        # Select fields

        # --------------------------------------------------



        columns = [

            "timestamp",

            "switch",

            "port",

            "rx_bytes",

            "tx_bytes",

            "rx_packets",

            "tx_packets",

            "rx_kbps",

            "tx_kbps",

            "total_kbps"

        ]



        available_columns = [

            column

            for column in columns

            if column in traffic_df.columns

        ]



        traffic_df = traffic_df[

            available_columns

        ]



        # --------------------------------------------------

        # Convert NaN to None

        # --------------------------------------------------



        traffic_df = traffic_df.astype(object).where(

            pd.notnull(traffic_df),

            None

        )



        records = traffic_df.to_dict(

            orient="records"

        )



        return jsonify({

            "status": "ACTIVE",

            "count": len(records),

            "records": records

        })



    except Exception as e:



        return jsonify({

            "error": str(e)

        }), 500





# ============================================================

# API 3: LIVE D1 DETECTION

# ============================================================



@app.route("/api/detection/live")

def detection_live():



    if not os.path.exists(ANALYTICS_FILE):



        return jsonify({

            "error": "Analytics file not found",

            "path": ANALYTICS_FILE

        }), 404



    try:



        df = pd.read_csv(ANALYTICS_FILE)



        if df.empty:



            return jsonify({

                "status": "NO_DATA",

                "count": 0,

                "records": []

            })



        detection_df = df[

            df["d1_prediction"].notna()

        ].copy()



        if detection_df.empty:



            return jsonify({

                "status": "NO_DETECTION_DATA",

                "count": 0,

                "records": []

            })



        # --------------------------------------------------

        # Convert timestamp

        # --------------------------------------------------



        detection_df["timestamp"] = pd.to_datetime(

            detection_df["timestamp"],

            errors="coerce"

        )



        detection_df = detection_df[

            detection_df["timestamp"].notna()

        ].copy()



        # --------------------------------------------------

        # Convert numeric fields

        # --------------------------------------------------



        numeric_columns = [

            "switch",

            "port",

            "d1_class",

            "normal_probability",

            "malicious_probability",

            "ip_proto",

            "src_port",

            "dst_port"

        ]



        for column in numeric_columns:



            if column in detection_df.columns:



                detection_df[column] = pd.to_numeric(

                    detection_df[column],

                    errors="coerce"

                )



        # --------------------------------------------------

        # Sort newest first

        # --------------------------------------------------



        detection_df = detection_df.sort_values(

            "timestamp",

            ascending=False

        )



        detection_df = detection_df.head(20).copy()



        detection_df["timestamp"] = (

            detection_df["timestamp"]

            .dt.strftime("%Y-%m-%d %H:%M:%S")

        )



        # --------------------------------------------------

        # Select fields

        # --------------------------------------------------



        columns = [

            "timestamp",

            "switch",

            "port",

            "d1_prediction",

            "d1_class",

            "normal_probability",

            "malicious_probability",

            "mitigation_status",

            "ipv4_src",

            "ipv4_dst",

            "ip_proto",

            "src_port",

            "dst_port"

        ]



        available_columns = [

            column

            for column in columns

            if column in detection_df.columns

        ]



        detection_df = detection_df[

            available_columns

        ]



        # --------------------------------------------------

        # Convert NaN to None

        # --------------------------------------------------



        detection_df = detection_df.astype(object).where(

            pd.notnull(detection_df),

            None

        )



        records = detection_df.to_dict(

            orient="records"

        )



        return jsonify({

            "status": "ACTIVE",

            "count": len(records),

            "records": records

        })



    except Exception as e:



        return jsonify({

            "error": str(e)

        }), 500





# ============================================================

# API 4: ATTACK RECORDS

# ============================================================



@app.route("/api/attacks")

def attacks():



    if not os.path.exists(ANALYTICS_FILE):



        return jsonify({

            "error": "Analytics file not found",

            "path": ANALYTICS_FILE

        }), 404



    try:



        df = pd.read_csv(ANALYTICS_FILE)



        if df.empty:



            return jsonify({

                "status": "NO_DATA",

                "count": 0,

                "records": []

            })



        # --------------------------------------------------

        # Select only malicious D1 records

        # --------------------------------------------------



        attack_df = df[

            df["d1_prediction"]

            .astype(str)

            .str.upper()

            == "MALICIOUS"

        ].copy()



        if attack_df.empty:



            return jsonify({

                "status": "NO_ATTACKS",

                "count": 0,

                "records": []

            })



        # --------------------------------------------------

        # Convert timestamp

        # --------------------------------------------------



        attack_df["timestamp"] = pd.to_datetime(

            attack_df["timestamp"],

            errors="coerce"

        )



        attack_df = attack_df[

            attack_df["timestamp"].notna()

        ].copy()



        # --------------------------------------------------

        # Convert numeric fields

        # --------------------------------------------------



        numeric_columns = [

            "switch",

            "port",

            "d1_class",

            "normal_probability",

            "malicious_probability",

            "ip_proto",

            "src_port",

            "dst_port"

        ]



        for column in numeric_columns:



            if column in attack_df.columns:



                attack_df[column] = pd.to_numeric(

                    attack_df[column],

                    errors="coerce"

                )



        # --------------------------------------------------

        # Sort newest attacks first

        # --------------------------------------------------



        attack_df = attack_df.sort_values(

            "timestamp",

            ascending=False

        )



        # --------------------------------------------------

        # Latest 50 attacks

        # --------------------------------------------------



        attack_df = attack_df.head(50).copy()



        attack_df["timestamp"] = (

            attack_df["timestamp"]

            .dt.strftime("%Y-%m-%d %H:%M:%S")

        )



        # --------------------------------------------------

        # Select attack fields

        # --------------------------------------------------



        columns = [

            "timestamp",

            "switch",

            "port",

            "d1_prediction",

            "d1_class",

            "malicious_probability",

            "mitigation_status",

            "ipv4_src",

            "ipv4_dst",

            "ip_proto",

            "src_port",

            "dst_port"

        ]



        available_columns = [

            column

            for column in columns

            if column in attack_df.columns

        ]



        attack_df = attack_df[

            available_columns

        ]



        # --------------------------------------------------

        # Convert NaN to None

        # --------------------------------------------------



        attack_df = attack_df.astype(object).where(

            pd.notnull(attack_df),

            None

        )



        records = attack_df.to_dict(

            orient="records"

        )



        return jsonify({

            "status": "ACTIVE",

            "count": len(records),

            "records": records

        })



    except Exception as e:



        return jsonify({

            "error": str(e)

        }), 500





# ============================================================

# API 5: TRAFFIC & BANDWIDTH ANALYTICS

# ============================================================



@app.route("/api/analytics/traffic")

def analytics_traffic():



    if not os.path.exists(ANALYTICS_FILE):



        return jsonify({

            "error": "Analytics file not found",

            "path": ANALYTICS_FILE

        }), 404



    try:



        df = pd.read_csv(ANALYTICS_FILE)



        if df.empty:



            return jsonify({

                "status": "NO_DATA",

                "count": 0,

                "records": []

            })



        # --------------------------------------------------

        # Select traffic records

        # --------------------------------------------------



        traffic_df = df[

            df["total_kbps"].notna()

        ].copy()



        if traffic_df.empty:



            return jsonify({

                "status": "NO_TRAFFIC_DATA",

                "count": 0,

                "records": []

            })



        # --------------------------------------------------

        # Convert numeric fields

        # --------------------------------------------------



        numeric_columns = [

            "switch",

            "port",

            "rx_bytes",

            "tx_bytes",

            "rx_packets",

            "tx_packets",

            "rx_kbps",

            "tx_kbps",

            "total_kbps"

        ]



        for column in numeric_columns:



            if column in traffic_df.columns:



                traffic_df[column] = pd.to_numeric(

                    traffic_df[column],

                    errors="coerce"

                )



        # --------------------------------------------------

        # Convert timestamp

        # --------------------------------------------------



        traffic_df["timestamp"] = pd.to_datetime(

            traffic_df["timestamp"],

            errors="coerce"

        )



        traffic_df = traffic_df[

            traffic_df["timestamp"].notna()

        ].copy()



        # --------------------------------------------------

        # Keep actual traffic records

        # --------------------------------------------------



        traffic_df = traffic_df[

            traffic_df["total_kbps"] > 0

        ].copy()



        if traffic_df.empty:



            return jsonify({

                "status": "NO_ACTIVE_TRAFFIC",

                "count": 0,

                "records": []

            })



        # --------------------------------------------------

        # Sort chronologically

        # --------------------------------------------------



        traffic_df = traffic_df.sort_values(

            "timestamp",

            ascending=True

        )



        # --------------------------------------------------

        # Latest 100 records

        # --------------------------------------------------



        traffic_df = traffic_df.tail(100).copy()



        # --------------------------------------------------

        # Format timestamp

        # --------------------------------------------------



        traffic_df["timestamp"] = (

            traffic_df["timestamp"]

            .dt.strftime("%Y-%m-%d %H:%M:%S")

        )



        # --------------------------------------------------

        # Select fields

        # --------------------------------------------------



        columns = [

            "timestamp",

            "switch",

            "port",

            "rx_bytes",

            "tx_bytes",

            "rx_packets",

            "tx_packets",

            "rx_kbps",

            "tx_kbps",

            "total_kbps"

        ]



        available_columns = [

            column

            for column in columns

            if column in traffic_df.columns

        ]



        traffic_df = traffic_df[

            available_columns

        ]



        # --------------------------------------------------

        # Convert NaN to None

        # --------------------------------------------------



        traffic_df = traffic_df.astype(object).where(

            pd.notnull(traffic_df),

            None

        )



        records = traffic_df.to_dict(

            orient="records"

        )



        # --------------------------------------------------

        # Analytics summary

        # --------------------------------------------------



        total_traffic = float(

            traffic_df["total_kbps"]

            .fillna(0)

            .sum()

        )



        peak_bandwidth = float(

            traffic_df["total_kbps"]

            .fillna(0)

            .max()

        )



        average_bandwidth = float(

            traffic_df["total_kbps"]

            .fillna(0)

            .mean()

        )



        return jsonify({

            "status": "ACTIVE",

            "count": len(records),

            "summary": {

                "total_traffic_kbps": round(

                    total_traffic,

                    2

                ),

                "peak_bandwidth_kbps": round(

                    peak_bandwidth,

                    2

                ),

                "average_bandwidth_kbps": round(

                    average_bandwidth,

                    2

                )

            },

            "records": records

        })



    except Exception as e:



        return jsonify({

            "error": str(e)

        }), 500





# ============================================================

# API 6: MODEL COMPARISON

# ============================================================



@app.route("/api/models/comparison")

def models_comparison():



    try:



        # --------------------------------------------------

        # Model metrics

        # --------------------------------------------------



        models = [



            # ============================

            # D1 MODELS

            # ============================



            {

                "dataset": "D1",

                "model": "DNN",

                "accuracy": 99.19,

                "precision": 98.57,

                "recall": 99.33,

                "f1_score": 98.95,

                "fpr": 0.89

            },



            {

                "dataset": "D1",

                "model": "CNN",

                "accuracy": 98.97,

                "precision": 98.26,

                "recall": 99.07,

                "f1_score": 98.66,

                "fpr": 1.09

            },



            {

                "dataset": "D1",

                "model": "LSTM",

                "accuracy": 92.94,

                "precision": 86.90,

                "recall": 90.75,

                "f1_score": 88.79,

                "fpr": 6.09

            },



            {

                "dataset": "D1",

                "model": "TabNet",

                "accuracy": 96.30,

                "precision": 93.85,

                "recall": 96.67,

                "f1_score": 95.24,

                "fpr": 3.93

            },





            # ============================

            # D2 MODELS

            # ============================



            {

                "dataset": "D2",

                "model": "DNN",

                "accuracy": 100.00,

                "precision": 100.00,

                "recall": 100.00,

                "f1_score": 100.00,

                "fpr": 0.00

            },



            {

                "dataset": "D2",

                "model": "CNN",

                "accuracy": 100.00,

                "precision": 100.00,

                "recall": 100.00,

                "f1_score": 100.00,

                "fpr": 0.00

            },



            {

                "dataset": "D2",

                "model": "TabNet",

                "accuracy": 100.00,

                "precision": 100.00,

                "recall": 100.00,

                "f1_score": 100.00,

                "fpr": 0.00

            }

        ]



        # --------------------------------------------------

        # Find best D1 model

        # --------------------------------------------------



        d1_models = [

            model

            for model in models

            if model["dataset"] == "D1"

        ]



        best_d1_model = max(

            d1_models,

            key=lambda model: model["f1_score"]

        )



        # --------------------------------------------------

        # Find best D2 model

        # --------------------------------------------------



        d2_models = [

            model

            for model in models

            if model["dataset"] == "D2"

        ]



        best_d2_model = max(

            d2_models,

            key=lambda model: model["f1_score"]

        )



        # --------------------------------------------------

        # Response

        # --------------------------------------------------



        return jsonify({

            "status": "ACTIVE",

            "models": models,

            "best_models": {

                "D1": best_d1_model,

                "D2": best_d2_model

            }

        })



    except Exception as e:



        return jsonify({

            "error": str(e)

        }), 500





# ============================================================


# ============================================================
# API 7: MITIGATION STATUS
# ============================================================

@app.route("/api/mitigation/status")
def mitigation_status():

    mitigation_csv = os.path.join(
        BASE_DIR,
        "results",
        "mitigation",
        "d2_offline_mitigation.csv"
    )

    try:

        # --------------------------------------------------
        # Check mitigation CSV
        # --------------------------------------------------

        if not os.path.exists(mitigation_csv):

            return jsonify({
                "status": "NO_DATA",
                "message": "D2 mitigation CSV file not found",
                "path": mitigation_csv
            }), 404

        # --------------------------------------------------
        # Load mitigation results
        # --------------------------------------------------

        df = pd.read_csv(
            mitigation_csv
        )

        if df.empty:

            return jsonify({
                "status": "NO_DATA",
                "summary": {
                    "total_samples": 0,
                    "normal_samples": 0,
                    "malicious_samples": 0,
                    "ddos_icmp": 0,
                    "ddos_tcp": 0,
                    "ddos_udp": 0,
                    "mitigation_needed": 0,
                    "mitigation_rate": 0.0
                },
                "records": []
            })

        # --------------------------------------------------
        # Normalize text columns
        # --------------------------------------------------

        df["attack_status"] = (
            df["attack_status"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        df["predicted_class"] = (
            df["predicted_class"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        # --------------------------------------------------
        # Total samples
        # --------------------------------------------------

        total_samples = len(df)

        # --------------------------------------------------
        # Normal traffic
        # --------------------------------------------------

        normal_samples = int(
            (
                df["attack_status"] == "NORMAL"
            ).sum()
        )

        # --------------------------------------------------
        # Malicious traffic
        # --------------------------------------------------

        malicious_samples = int(
            (
                df["attack_status"] == "MALICIOUS"
            ).sum()
        )

        # --------------------------------------------------
        # DDoS attack types
        # --------------------------------------------------

        ddos_icmp = int(
            (
                df["predicted_class"] == "DDOS_ICMP"
            ).sum()
        )

        ddos_tcp = int(
            (
                df["predicted_class"] == "DDOS_TCP"
            ).sum()
        )

        ddos_udp = int(
            (
                df["predicted_class"] == "DDOS_UDP"
            ).sum()
        )

        # --------------------------------------------------
        # Mitigation required
        #
        # Every malicious DDoS sample requires mitigation.
        # --------------------------------------------------

        mitigation_needed = malicious_samples

        # --------------------------------------------------
        # Mitigation rate
        # --------------------------------------------------

        if total_samples > 0:

            mitigation_rate = (
                mitigation_needed
                / total_samples
            ) * 100

        else:

            mitigation_rate = 0.0

        # --------------------------------------------------
        # Detailed records
        # --------------------------------------------------

        records_df = df.head(50).copy()

        # Convert NaN to None
        records_df = records_df.astype(
            object
        ).where(
            pd.notnull(records_df),
            None
        )

        records = records_df.to_dict(
            orient="records"
        )

        # --------------------------------------------------
        # Response
        # --------------------------------------------------

        return jsonify({

            "status": "ACTIVE",

            "summary": {

                "total_samples": total_samples,

                "normal_samples": normal_samples,

                "malicious_samples": malicious_samples,

                "ddos_icmp": ddos_icmp,

                "ddos_tcp": ddos_tcp,

                "ddos_udp": ddos_udp,

                "mitigation_needed": mitigation_needed,

                "mitigation_rate": round(
                    mitigation_rate,
                    2
                )
            },

            "records": records

        })

    except Exception as e:

        return jsonify({
            "status": "ERROR",
            "error": str(e)
        }), 500

# START SERVER

# ============================================================



if __name__ == "__main__":



    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )