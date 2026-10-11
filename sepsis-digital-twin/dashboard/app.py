import os
import sys

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from dash import Dash, html, dcc, dash_table, Input, Output, State
from dash.exceptions import PreventUpdate


# ---------------------------------------------------------
# Make project root available
# ---------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ---------------------------------------------------------
# Import your prediction function
# ---------------------------------------------------------

from src.inference.predict_patient import predict_patient


# ---------------------------------------------------------
# Create Dash application
# ---------------------------------------------------------

app = Dash(__name__)

app.title = "Sepsis Digital Twin"


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def card(title, value, card_id=None):
    return html.Div(
        [
            html.Div(title, className="card-title"),
            html.Div(
                value,
                id=card_id,
                className="card-value"
            )
        ],
        className="card"
    )


# ---------------------------------------------------------
# Layout
# ---------------------------------------------------------

app.layout = html.Div(
    [

        # Header
        html.Div(
            [
                html.H1("SEPSIS DIGITAL TWIN"),
                html.P(
                    "Patient-level AI risk monitoring and explainability"
                )
            ],
            className="header"
        ),

        # Patient input
        html.Div(
            [
                html.Label("Enter Patient ID"),

                dcc.Input(
                    id="patient-input",
                    type="text",
                    value="p100023",
                    placeholder="Example: p100023",
                    className="patient-input"
                ),

                html.Button(
                    "Run Prediction",
                    id="run-button",
                    n_clicks=0,
                    className="run-button"
                ),

                html.Div(
                    id="status-message",
                    className="status"
                )
            ],
            className="input-section"
        ),

        # KPI cards
        html.Div(
            [
                card(
                    "Current Risk",
                    "--",
                    "current-risk"
                ),

                card(
                    "Alert Status",
                    "--",
                    "alert-status"
                ),

                card(
                    "First Alert",
                    "--",
                    "first-alert"
                ),

                card(
                    "Warning Time",
                    "--",
                    "warning-time"
                ),
            ],
            className="cards"
        ),

        # Risk graph
        html.Div(
            [
                html.H2("Sepsis Risk Trajectory"),

                dcc.Graph(
                    id="risk-graph"
                )
            ],
            className="section"
        ),

        # Vital graph
        html.Div(
            [
                html.H2("Patient Physiological Trends"),

                dcc.Dropdown(
                    id="feature-dropdown",
                    placeholder="Select a vital/lab feature"
                ),

                dcc.Graph(
                    id="vital-graph"
                )
            ],
            className="section"
        ),

        # SHAP
        html.Div(
            [
                html.H2("Model Explanation — Top Features"),

                dcc.Graph(
                    id="shap-graph"
                )
            ],
            className="section"
        ),

        # Prediction table
        html.Div(
            [
                html.H2("Hourly Patient Predictions"),

                dash_table.DataTable(
                    id="prediction-table",

                    columns=[],

                    data=[],

                    page_size=15,

                    sort_action="native",

                    filter_action="native",

                    style_table={
                        "overflowX": "auto"
                    },

                    style_cell={
                        "textAlign": "center",
                        "padding": "10px"
                    },

                    style_header={
                        "fontWeight": "bold"
                    }
                )
            ],
            className="section"
        )
    ],

    className="dashboard"
)


# ---------------------------------------------------------
# Run prediction
# ---------------------------------------------------------

@app.callback(
    [
        Output("current-risk", "children"),
        Output("alert-status", "children"),
        Output("first-alert", "children"),
        Output("warning-time", "children"),
        Output("risk-graph", "figure"),
        Output("feature-dropdown", "options"),
        Output("feature-dropdown", "value"),
        Output("vital-graph", "figure"),
        Output("shap-graph", "figure"),
        Output("prediction-table", "columns"),
        Output("prediction-table", "data"),
        Output("status-message", "children"),
    ],

    Input("run-button", "n_clicks"),

    State("patient-input", "value"),

    prevent_initial_call=True
)
def update_dashboard(n_clicks, patient_id):

    if not patient_id:
        raise PreventUpdate

    patient_id = patient_id.strip()

    try:

        # ---------------------------------------------
        # Run your existing ML pipeline
        # ---------------------------------------------

        result = predict_patient(patient_id)

        df = result["predictions"]

        probs = result["probs"]

        threshold = result["threshold"]

        first_alert_hour = result["first_alert_hour"]

        actual_onset_hour = result["actual_onset_hour"]

        warning_time = result["warning_time"]

        top_features = result["top_features"]


        # ---------------------------------------------
        # Current risk
        # ---------------------------------------------

        current_risk = float(probs[-1])

        risk_text = f"{current_risk * 100:.2f}%"


        # ---------------------------------------------
        # Alert status
        # ---------------------------------------------

        has_alert = first_alert_hour is not None

        if has_alert:
            alert_text = "🔴 HIGH RISK"
        else:
            alert_text = "🟢 NO ALERT"


        # ---------------------------------------------
        # First alert
        # ---------------------------------------------

        if first_alert_hour is not None:
            first_alert_text = f"Hour {first_alert_hour}"
        else:
            first_alert_text = "No alert"


        # ---------------------------------------------
        # Warning time
        # ---------------------------------------------

        if warning_time is not None:
            warning_text = f"{warning_time} hours"
        else:
            warning_text = "N/A"


        # ---------------------------------------------
        # Risk trajectory
        # ---------------------------------------------

        fig_risk = go.Figure()

        fig_risk.add_trace(
            go.Scatter(
                x=df["ICULOS"],
                y=df["PredictedRisk"],
                mode="lines+markers",
                name="Predicted Risk"
            )
        )

        fig_risk.add_hline(
            y=threshold,
            line_dash="dash",
            annotation_text=f"Alert threshold: {threshold:.2f}"
        )

        # Ground truth
        if "SepsisLabel" in df.columns:

            fig_risk.add_trace(
                go.Scatter(
                    x=df["ICULOS"],
                    y=df["SepsisLabel"],
                    mode="lines",
                    name="Ground Truth"
                )
            )


        # First alert marker
        if first_alert_hour is not None:

            alert_rows = df[
                df["ICULOS"] == first_alert_hour
            ]

            if not alert_rows.empty:

                alert_risk = alert_rows["PredictedRisk"].iloc[0]

                fig_risk.add_trace(
                    go.Scatter(
                        x=[first_alert_hour],
                        y=[alert_risk],
                        mode="markers",
                        marker=dict(
                            size=14
                        ),
                        name="First Alert"
                    )
                )


        fig_risk.update_layout(
            title=f"Patient {patient_id} — Sepsis Risk",
            xaxis_title="ICU Stay Hour",
            yaxis_title="Risk Probability",
            yaxis=dict(range=[0, 1]),
            hovermode="x unified"
        )


        # ---------------------------------------------
        # Feature dropdown
        # ---------------------------------------------

        excluded = {
            "PatientID",
            "SepsisLabel",
            "PredictedRisk",
            "SepsisAlert"
        }

        feature_options = [
            {"label": col, "value": col}
            for col in df.columns
            if col not in excluded
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        if feature_options:

            selected_feature = feature_options[0]["value"]

        else:

            selected_feature = None


        # ---------------------------------------------
        # Vital graph
        # ---------------------------------------------

        if selected_feature:

            fig_vital = px.line(
                df,
                x="ICULOS",
                y=selected_feature,
                markers=True,
                title=f"{selected_feature} over ICU time"
            )

            fig_vital.update_layout(
                xaxis_title="ICU Stay Hour",
                yaxis_title=selected_feature
            )

        else:

            fig_vital = go.Figure()


        # ---------------------------------------------
        # SHAP graph
        # ---------------------------------------------

        shap_df = pd.DataFrame(
            top_features,
            columns=[
                "Feature",
                "Value",
                "SHAP"
            ]
        )

        shap_df = shap_df.sort_values(
            "SHAP",
            ascending=True
        )

        fig_shap = px.bar(
            shap_df,
            x="SHAP",
            y="Feature",
            orientation="h",
            title="Top 5 Features Contributing to Risk"
        )

        fig_shap.update_layout(
            xaxis_title="SHAP Contribution",
            yaxis_title="Feature"
        )


        # ---------------------------------------------
        # Prediction table
        # ---------------------------------------------

        table_columns = [
            {
                "name": col,
                "id": col
            }
            for col in [
                "PatientID",
                "ICULOS",
                "SepsisLabel",
                "PredictedRisk",
                "SepsisAlert"
            ]
            if col in df.columns
        ]

        table_data = df[
            [col["id"] for col in table_columns]
        ].copy()

        table_data["PredictedRisk"] = (
            table_data["PredictedRisk"]
            .round(4)
        )

        table_data = table_data.to_dict("records")


        return (
            risk_text,
            alert_text,
            first_alert_text,
            warning_text,
            fig_risk,
            feature_options,
            selected_feature,
            fig_vital,
            fig_shap,
            table_columns,
            table_data,
            f"Prediction completed for {patient_id}"
        )


    except FileNotFoundError:

        empty = go.Figure()

        return (
            "--",
            "Patient not found",
            "--",
            "--",
            empty,
            [],
            None,
            empty,
            empty,
            [],
            [],
            f"❌ Patient file not found: {patient_id}"
        )


    except Exception as e:

        empty = go.Figure()

        return (
            "--",
            "Error",
            "--",
            "--",
            empty,
            [],
            None,
            empty,
            empty,
            [],
            [],
            f"❌ Error: {str(e)}"
        )


# ---------------------------------------------------------
# Start server
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=8050
    )