import { useEffect, useRef, useState } from "react";
import "./App.css";

type PatientRow = Record<string, string | number | null>;

type ShapFeature = {
  feature: string;
  value: number;
  shap_value: number;
};

type ShapExplanation = {
  available: boolean;
  explained_output?: string;
  message?: string;
  positive_contributors?: ShapFeature[];
  negative_contributors?: ShapFeature[];
  top_features?: ShapFeature[];
};

type Prediction = {
  risk: number;
  risk_percentage: number;
  threshold: number;
  alert: boolean;
  severity: string;
  hour: number | null;
  shap?: ShapExplanation;
};

type HistoryEntry = {
  hour: number;
  risk: number;
  severity: string;
  alert: boolean;
};

const API_URL = "http://localhost:8000";
const SIMULATION_INTERVAL = 8000;

function parsePSV(text: string): PatientRow[] {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);

  if (lines.length < 2) {
    throw new Error("The patient file contains no data rows.");
  }

  const headers = lines[0].split("|").map((header) => header.trim());

  return lines.slice(1).map((line) => {
    const values = line.split("|");
    const row: PatientRow = {};

    headers.forEach((header, index) => {
      const value = values[index]?.trim() ?? "";

      if (
        value === "" ||
        ["nan", "null", "none"].includes(value.toLowerCase())
      ) {
        row[header] = null;
      } else {
        const numberValue = Number(value);
        row[header] = Number.isFinite(numberValue) ? numberValue : value;
      }
    });

    return row;
  });
}

function getSeverityColor(severity: string): string {
  switch (severity) {
    case "LOW":
      return "#16a34a";
    case "MODERATE":
      return "#ca8a04";
    case "HIGH":
      return "#ea580c";
    case "CRITICAL":
      return "#dc2626";
    default:
      return "#64748b";
  }
}

function formatVital(row: PatientRow | undefined, key: string): string {
  if (!row || row[key] === null || row[key] === undefined) {
    return "--";
  }

  const value = row[key];

  return typeof value === "number" ? value.toFixed(1) : String(value);
}

function formatFeatureName(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\s+/g, " ")
    .trim();
}

function formatShapValue(value: number): string {
  return `${value > 0 ? "+" : ""}${value.toFixed(4)}`;
}

export default function App() {
  const [patientData, setPatientData] = useState<PatientRow[]>([]);
  const [patientId, setPatientId] = useState("p000001");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [countdown, setCountdown] = useState(8);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const [trajectoryAvailable, setTrajectoryAvailable] = useState(false);
  const [trajectoryError, setTrajectoryError] = useState("");

  const timerRef = useRef<number | null>(null);
  const busyRef = useRef(false);
  const indexRef = useRef(0);
  const dataRef = useRef<PatientRow[]>([]);
  const predictionRef = useRef<Prediction | null>(null);
  const runningRef = useRef(false);

  const currentRow = patientData[currentIndex];

  const completed =
    patientData.length > 0 &&
    currentIndex >= patientData.length - 1 &&
    prediction !== null;

  const severityColor = prediction
    ? getSeverityColor(prediction.severity)
    : "#64748b";

  const shapData = prediction?.shap;

  useEffect(() => {
    runningRef.current = running;
  }, [running]);

  useEffect(() => {
    dataRef.current = patientData;
  }, [patientData]);

  useEffect(() => {
    predictionRef.current = prediction;
  }, [prediction]);

  function clearTimer() {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  async function loadPatientFile(file: File) {
    try {
      setError("");
      setRunning(false);
      runningRef.current = false;
      clearTimer();

      const text = await file.text();
      const rows = parsePSV(text);

      setPatientData(rows);
      dataRef.current = rows;

      setPatientId(file.name.replace(/\.psv$/i, ""));

      setCurrentIndex(0);
      indexRef.current = 0;

      setPrediction(null);
      predictionRef.current = null;

      setHistory([]);
      setCountdown(8);
      setElapsedSeconds(0);

      setTrajectoryAvailable(false);
      setTrajectoryError("");

      busyRef.current = false;
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to load patient data.",
      );
    }
  }

  async function predictHour(index: number) {
    const rows = dataRef.current;

    if (index < 0 || index >= rows.length) {
      setRunning(false);
      runningRef.current = false;
      clearTimer();
      return;
    }

    if (busyRef.current) {
      return;
    }

    busyRef.current = true;
    setLoading(true);
    setError("");

    try {
      // Send all observations through the current simulated hour.
      const observations = rows.slice(0, index + 1);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          patient_data: observations,
        }),
      });

      if (!response.ok) {
        const detail = await response.text();

        throw new Error(`Prediction API error (${response.status}): ${detail}`);
      }

      const result: Prediction = await response.json();

      setCurrentIndex(index);
      indexRef.current = index;

      setPrediction(result);
      predictionRef.current = result;

      setHistory((previous) => {
        const entry: HistoryEntry = {
          hour: result.hour ?? index + 1,
          risk: result.risk_percentage,
          severity: result.severity,
          alert: result.alert,
        };

        // Avoid duplicate entries if the same hour is requested again.
        const withoutCurrentHour = previous.filter(
          (item) => item.hour !== entry.hour,
        );

        return [...withoutCurrentHour, entry];
      });

      setElapsedSeconds(index * (SIMULATION_INTERVAL / 1000));

      if (index >= rows.length - 1) {
        setRunning(false);
        runningRef.current = false;
        clearTimer();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");

      setRunning(false);
      runningRef.current = false;
      clearTimer();
    } finally {
      busyRef.current = false;
      setLoading(false);
    }
  }

  async function startSimulation() {
    if (dataRef.current.length === 0) {
      setError("Please upload a patient PSV file first.");
      return;
    }

    if (
      indexRef.current >= dataRef.current.length - 1 &&
      predictionRef.current !== null
    ) {
      setError("Simulation completed. Reset to run it again.");
      return;
    }

    setError("");
    setRunning(true);
    runningRef.current = true;

    if (predictionRef.current === null) {
      await predictHour(0);
    }
  }

  function pauseSimulation() {
    setRunning(false);
    runningRef.current = false;
    clearTimer();
  }

  function resetSimulation() {
    pauseSimulation();

    setCurrentIndex(0);
    indexRef.current = 0;

    setPrediction(null);
    predictionRef.current = null;

    setHistory([]);
    setCountdown(8);
    setElapsedSeconds(0);

    setTrajectoryAvailable(false);
    setTrajectoryError("");
    setError("");

    busyRef.current = false;
  }

  // Simulation timer.
  useEffect(() => {
    if (!running) {
      clearTimer();
      return;
    }

    setCountdown(8);

    timerRef.current = window.setInterval(() => {
      setCountdown((previous) => {
        if (previous <= 1) {
          if (!busyRef.current) {
            const nextIndex = indexRef.current + 1;

            if (nextIndex < dataRef.current.length) {
              void predictHour(nextIndex);
            } else {
              setRunning(false);
              runningRef.current = false;
            }
          }

          return 8;
        }

        return previous - 1;
      });
    }, 1000);

    return clearTimer;
  }, [running]);

  // Check whether the graph endpoint is available.
  useEffect(() => {
    let cancelled = false;

    async function checkTrajectory() {
      try {
        const response = await fetch(`${API_URL}/risk-trajectory`, {
          method: "HEAD",
        });

        if (!cancelled) {
          setTrajectoryAvailable(response.ok);

          if (!response.ok) {
            setTrajectoryError(
              "The risk trajectory graph is not available yet.",
            );
          } else {
            setTrajectoryError("");
          }
        }
      } catch {
        if (!cancelled) {
          setTrajectoryAvailable(false);
          setTrajectoryError(
            "Cannot connect to the graph endpoint. Check that FastAPI is running.",
          );
        }
      }
    }

    void checkTrajectory();

    return () => {
      cancelled = true;
    };
  }, []);

  const positiveContributors =
    shapData?.available && shapData.positive_contributors
      ? shapData.positive_contributors
      : [];

  const negativeContributors =
    shapData?.available && shapData.negative_contributors
      ? shapData.negative_contributors
      : [];

  const topFeatures =
    shapData?.available && shapData.top_features ? shapData.top_features : [];

  return (
    <main className="dashboard">
      <header className="topbar">
        <div>
          <h1>Sepsis Digital Twin</h1>
          <p>
            Simulated ICU patient monitoring, XGBoost risk prediction, and SHAP
            explainability
          </p>
        </div>

        <div className="system-status">
          <span
            className="status-dot"
            style={{
              backgroundColor: error
                ? "#dc2626"
                : running
                  ? "#16a34a"
                  : "#64748b",
            }}
          />

          {loading
            ? "Predicting..."
            : running
              ? "Simulation Running"
              : completed
                ? "Simulation Completed"
                : "Simulation Paused"}
        </div>
      </header>

      {/* Patient and simulation information */}
      <section className="patient-panel">
        <div>
          <label htmlFor="patient-id">Patient ID</label>
          <input
            id="patient-id"
            value={patientId}
            onChange={(event) => setPatientId(event.target.value)}
          />
        </div>

        <div>
          <label>Simulation Hour</label>
          <strong>
            {prediction?.hour ?? (patientData.length ? currentIndex + 1 : 0)}
          </strong>
        </div>

        <div>
          <label>Available Observations</label>
          <strong>{patientData.length}</strong>
        </div>

        <div>
          <label>Simulation Interval</label>
          <strong>8 seconds</strong>
        </div>

        <div>
          <label>Elapsed Simulation Time</label>
          <strong>{elapsedSeconds}s</strong>
        </div>
      </section>

      {/* Controls */}
      <section className="controls">
        <label className="upload-button">
          Upload Patient PSV
          <input
            type="file"
            accept=".psv,.txt"
            onChange={(event) => {
              const file = event.target.files?.[0];

              if (file) {
                void loadPatientFile(file);
              }

              event.currentTarget.value = "";
            }}
          />
        </label>

        {!running ? (
          <button
            className="primary-button"
            onClick={() => void startSimulation()}
            disabled={loading || completed}
          >
            ▶ Start Simulation
          </button>
        ) : (
          <button className="pause-button" onClick={pauseSimulation}>
            ⏸ Pause
          </button>
        )}

        <button className="secondary-button" onClick={resetSimulation}>
          ↻ Reset
        </button>

        <div className="countdown">
          {running ? `Next prediction in ${countdown}s` : "Waiting"}
        </div>
      </section>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Main risk metrics */}
      <section className="metrics-grid">
        <article className="metric-card">
          <span>Current Sepsis Risk</span>
          <h2 style={{ color: severityColor }}>
            {prediction ? `${prediction.risk_percentage.toFixed(2)}%` : "--"}
          </h2>
          <p>Latest XGBoost prediction</p>
        </article>

        <article className="metric-card">
          <span>Risk Classification</span>
          <h2 style={{ color: severityColor }}>
            {prediction?.severity ?? "--"}
          </h2>
          <p>Configured risk bands</p>
        </article>

        <article className="metric-card">
          <span>Model Alert</span>
          <h2
            style={{
              color: prediction?.alert ? "#dc2626" : "#16a34a",
            }}
          >
            {prediction ? (prediction.alert ? "ALERT" : "NO ALERT") : "--"}
          </h2>
          <p>
            Threshold:{" "}
            {prediction ? `${(prediction.threshold * 100).toFixed(2)}%` : "--"}
          </p>
        </article>

        <article className="metric-card">
          <span>Successful Predictions</span>
          <h2>{history.length}</h2>
          <p>Processed simulation observations</p>
        </article>
      </section>

      {/* Patient vitals and history */}
      <section className="content-grid">
        <article className="panel">
          <h2>Current Patient Vitals</h2>
          <p className="panel-description">
            Measurements from the current simulated ICU hour
          </p>

          <div className="vitals-grid">
            {[
              { key: "HR", label: "Heart Rate", unit: "bpm" },
              { key: "O2Sat", label: "Oxygen Saturation", unit: "%" },
              { key: "Temp", label: "Temperature", unit: "°C" },
              { key: "SBP", label: "Systolic BP", unit: "mmHg" },
              { key: "DBP", label: "Diastolic BP", unit: "mmHg" },
              { key: "Resp", label: "Respiratory Rate", unit: "/min" },
              { key: "MAP", label: "Mean Arterial Pressure", unit: "mmHg" },
              { key: "ICULOS", label: "ICU Length of Stay", unit: "hours" },
            ].map((vital) => (
              <div className="vital-card" key={vital.key}>
                <span>{vital.label}</span>
                <strong>{formatVital(currentRow, vital.key)}</strong>
                <small>{vital.unit}</small>
              </div>
            ))}
          </div>

          <p className="note">
            Values come from the uploaded observations. Missing measurements are
            displayed as --.
          </p>
        </article>

        <article className="panel">
          <h2>Prediction History</h2>
          <p className="panel-description">
            Sepsis risk across simulated ICU hours
          </p>

          {history.length === 0 ? (
            <div className="empty-state">
              Start the simulation to view predictions.
            </div>
          ) : (
            <div className="history-list">
              {history
                .slice()
                .reverse()
                .map((entry, index) => (
                  <div className="history-row" key={`${entry.hour}-${index}`}>
                    <div>
                      <strong>Hour {entry.hour}</strong>
                      <span
                        className="severity-badge"
                        style={{
                          color: getSeverityColor(entry.severity),
                        }}
                      >
                        {entry.severity}
                      </span>
                    </div>

                    <div className="history-risk">
                      <strong>{entry.risk.toFixed(2)}%</strong>
                      {entry.alert && <span className="alert-tag">ALERT</span>}
                    </div>
                  </div>
                ))}
            </div>
          )}
        </article>
      </section>

      {/* SHAP explanations */}
      <section className="panel shap-panel">
        <div className="section-heading">
          <div>
            <h2>SHAP Explainability</h2>
            <p className="panel-description">
              Features contributing to the latest model output
            </p>
          </div>

          <span
            className="severity-badge"
            style={{
              color: shapData?.available ? "#16a34a" : "#64748b",
            }}
          >
            {shapData?.available ? "Explanation Available" : "Waiting for SHAP"}
          </span>
        </div>

        {!prediction ? (
          <div className="empty-state">
            Run the simulation to calculate feature contributions.
          </div>
        ) : !shapData?.available ? (
          <div className="empty-state">
            {shapData?.message ??
              "SHAP data was not returned by the prediction API."}
          </div>
        ) : (
          <>
            <p className="note">
              Positive SHAP values push the model output toward the sepsis
              class. Negative values push it away. SHAP values explain the model
              output and are not direct percentage-point changes in risk.
            </p>

            <div className="shap-grid">
              <div className="shap-card">
                <h3>Risk-Increasing Contributions</h3>
                <p className="panel-description">Positive SHAP values</p>

                {positiveContributors.length === 0 ? (
                  <div className="empty-state">
                    No positive contributions for this observation.
                  </div>
                ) : (
                  <div className="shap-list">
                    {positiveContributors.map((feature) => (
                      <div className="shap-row" key={feature.feature}>
                        <div className="shap-feature-info">
                          <strong>{formatFeatureName(feature.feature)}</strong>
                          <span>Value: {feature.value.toFixed(3)}</span>
                        </div>

                        <strong className="shap-positive">
                          {formatShapValue(feature.shap_value)}
                        </strong>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="shap-card">
                <h3>Risk-Reducing Contributions</h3>
                <p className="panel-description">Negative SHAP values</p>

                {negativeContributors.length === 0 ? (
                  <div className="empty-state">
                    No negative contributions for this observation.
                  </div>
                ) : (
                  <div className="shap-list">
                    {negativeContributors.map((feature) => (
                      <div className="shap-row" key={feature.feature}>
                        <div className="shap-feature-info">
                          <strong>{formatFeatureName(feature.feature)}</strong>
                          <span>Value: {feature.value.toFixed(3)}</span>
                        </div>

                        <strong className="shap-negative">
                          {formatShapValue(feature.shap_value)}
                        </strong>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="shap-card shap-top-features">
              <h3>Top Features by Absolute SHAP Value</h3>

              {topFeatures.map((feature) => (
                <div className="shap-row" key={feature.feature}>
                  <div className="shap-feature-info">
                    <strong>{formatFeatureName(feature.feature)}</strong>
                    <span>Feature value: {feature.value.toFixed(3)}</span>
                  </div>

                  <strong
                    className={
                      feature.shap_value >= 0
                        ? "shap-positive"
                        : "shap-negative"
                    }
                  >
                    {formatShapValue(feature.shap_value)}
                  </strong>
                </div>
              ))}
            </div>
          </>
        )}
      </section>

      {/* Risk trajectory graph */}
      <section className="panel trajectory-panel">
        <div className="section-heading">
          <div>
            <h2>Risk Trajectory Graph</h2>
            <p className="panel-description">
              Visual reference generated by the project's inference pipeline
            </p>
          </div>

          <span
            className="severity-badge"
            style={{
              color: trajectoryAvailable ? "#16a34a" : "#64748b",
            }}
          >
            {trajectoryAvailable ? "Graph Available" : "Graph Status Unknown"}
          </span>
        </div>

        {trajectoryAvailable ? (
          <img
            className="trajectory-image"
            src={`${API_URL}/risk-trajectory`}
            alt="Sepsis risk trajectory across ICU hours"
            onError={() => {
              setTrajectoryAvailable(false);
              setTrajectoryError("Unable to load the risk trajectory image.");
            }}
          />
        ) : (
          <div className="empty-state">
            {trajectoryError ||
              "The graph will appear when the backend graph endpoint is available."}
          </div>
        )}

        <p className="note">
          This endpoint displays the saved PNG graph. It will not automatically
          redraw the graph for each uploaded patient unless the backend is
          updated to generate a patient-specific trajectory.
        </p>
      </section>

      {/* Progress */}
      <section className="panel progress-panel">
        <div className="progress-heading">
          <h2>Simulation Progress</h2>

          <strong>
            {patientData.length
              ? `${Math.min(
                  currentIndex + (prediction ? 1 : 0),
                  patientData.length,
                )} / ${patientData.length}`
              : "0 / 0"}
          </strong>
        </div>

        <div className="progress-track">
          <div
            className="progress-fill"
            style={{
              width: patientData.length
                ? `${
                    (Math.min(
                      currentIndex + (prediction ? 1 : 0),
                      patientData.length,
                    ) /
                      patientData.length) *
                    100
                  }%`
                : "0%",
            }}
          />
        </div>

        <p className="note">
          The simulation processes recorded observations. Each successful
          prediction updates the risk, history, and SHAP explanation.
        </p>
      </section>

      <section className="shap-section">
        <h2>Explainable AI — SHAP Analysis</h2>

        <div className="shap-grid">
          <div className="shap-card">
            <h3>SHAP Feature Importance</h3>
            <p>
              Features ranked by their overall contribution to model
              predictions.
            </p>

            <img
              src="/api/images/shap-importance"
              alt="SHAP feature importance plot"
              className="shap-image"
            />
          </div>

          <div className="shap-card">
            <h3>SHAP Summary Plot</h3>
            <p>Shows how patient features influence the model's predictions.</p>

            <img
              src="/api/images/shap-summary"
              alt="SHAP summary plot"
              className="shap-image"
            />
          </div>
        </div>
      </section>

      <footer>
        Sepsis Digital Twin · XGBoost · SHAP Explainability · FastAPI
      </footer>
    </main>
  );
}
