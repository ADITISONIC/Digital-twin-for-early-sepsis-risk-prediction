# Sepsis Digital Twin — ICU Monitoring Frontend

This frontend is designed around the existing project backend/model contract.

## Two demonstration modes

### Dataset Mode
Uses the existing Hospital B PhysioNet patient files through the backend:
- `GET /api/patients`
- `GET /api/patients/{patient_id}`
- `POST /api/predict`

This is the research/evaluation view.

### ICU Simulation Mode
Creates a **new virtual patient** in the browser. No manual vital entry is required. The simulator continuously generates a new patient's measurements and sends the evolving history through the same `/api/predict` endpoint.

Scenarios:
- Stable
- Gradual deterioration
- Rapid deterioration
- Recovery

The simulation does not bypass the ML model. Each generated state is sent to the backend, which runs the project's existing preprocessing, dynamic feature engineering, XGBoost model and SHAP explanation.

## Alerts

The UI shows two concepts separately:
1. The model's actual research decision threshold returned by the backend.
2. Demonstration alert bands (`LOW RISK`, `SUSPECT / MONITOR`, `HIGH RISK`, `EMERGENCY ALERT`).

These alert labels are **not clinical thresholds** and must not be presented as validated medical alarm rules.

## Run

From the project root:

```bash
pip install -r api/requirements.txt
uvicorn api.app:app --reload --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://localhost:8000`.

## Backend compatibility

The included `api/app.py` is a minimal FastAPI adapter that loads the existing `models/xgboost_model.pkl`, uses the project's `CausalPreprocessor` and `build_dynamic_features_pipeline`, and exposes the frontend endpoints. Place `api/` at the project root next to `src/`, `models/`, `data/`, and `config.yaml`.
