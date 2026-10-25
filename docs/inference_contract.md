# Model inference contract

- Load the trusted local pipeline with `student_performance.models.predict.load_model`.
- Load the matching `models/metadata.json` manifest for `raw_predictor_names`.
- Send raw features as a non-empty `pandas.DataFrame` containing exactly those
  column names in the recorded order. Do not include `Exam_Score`.
- The saved pipeline owns imputation, scaling, and one-hot encoding. Missing raw
  predictor values are handled by its fitted imputers; unseen categories are ignored
  by the fitted `OneHotEncoder(handle_unknown="ignore")`.
- Use `predict_from_dataframe` to enforce schema/order and receive a finite 1-D
  NumPy array. Missing columns, extra columns, duplicate names, or a changed order
  raise an actionable error.
- Keep the model artifact and metadata from the same training run together. Only
  load artifacts produced by this project or another trusted source; Joblib uses
  Python serialization and must not load untrusted files.
- Predictions are estimates, not guaranteed outcomes or causal explanations. No
  score clipping is applied.

## HTTP API

The FastAPI entry point is `student_performance.api.main:app`; service operations,
configuration, and request/response models live in `api/service.py`, `api/settings.py`,
and `api/schemas.py`. The versioned routes are registered in `api/main.py`. Install
declared dependencies into the existing environment only when creating a new one:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the local backend from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn student_performance.api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Interactive documentation is available at `http://127.0.0.1:8000/docs`; OpenAPI JSON
is at `http://127.0.0.1:8000/openapi.json`.
All application routes use the `/api/v1` prefix. `POST /api/v1/predict` accepts
exactly the 19 input features declared in the schema below; every key is required,
but a value may be `null` so the saved pipeline can impute it. Numeric fields must
be finite numbers, categorical fields must be strings, and unknown category strings
are accepted by the trained encoder. Extra keys (including `Exam_Score`) are
rejected. API errors use an `{ "error": { "code": "...", "message": "..." } }`
envelope.

Example request using a real dataset feature row:

```json
{
  "Hours_Studied": 23,
  "Attendance": 84,
  "Parental_Involvement": "Low",
  "Access_to_Resources": "High",
  "Extracurricular_Activities": "No",
  "Sleep_Hours": 7,
  "Previous_Scores": 73,
  "Motivation_Level": "Low",
  "Internet_Access": "Yes",
  "Tutoring_Sessions": 0,
  "Family_Income": "Low",
  "Teacher_Quality": "Medium",
  "School_Type": "Public",
  "Peer_Influence": "Positive",
  "Physical_Activity": 3,
  "Learning_Disabilities": "No",
  "Parental_Education_Level": "High School",
  "Distance_from_Home": "Near",
  "Gender": "Male"
}
```

The saved model returned this response for that real input:

```json
{
  "target_name": "Exam_Score",
  "predicted_score": 67.12516675552172,
  "is_estimate": true,
  "disclaimer": "This is an estimate for educational decision support, not a guaranteed academic outcome or definitive judgment."
}
```

Routes: `GET /api/v1/health`, `GET /api/v1/model-info`,
`GET /api/v1/analytics/summary`, `GET /api/v1/analytics/distributions?feature=...`,
`GET /api/v1/models/metrics`, `GET /api/v1/models/feature-importance`, and
`POST /api/v1/predict`.

Environment variables (all optional):

| Variable | Default | Purpose |
| --- | --- | --- |
| `STUDENT_PERFORMANCE_MODEL_PATH` | `models/best_model.joblib` | Trusted serialized fitted pipeline |
| `STUDENT_PERFORMANCE_METADATA_PATH` | `models/metadata.json` | Matching model manifest and feature order |
| `STUDENT_PERFORMANCE_DATASET_PATH` | `Dataset/StudentPerformanceFactors.csv` | Dataset for live analytics |
| `STUDENT_PERFORMANCE_MODEL_COMPARISON_PATH` | `reports/model_comparison.csv` | Training-only CV comparison |
| `STUDENT_PERFORMANCE_METRICS_PATH` | `reports/final_metrics.json` | Holdout metrics from training |
| `STUDENT_PERFORMANCE_FEATURE_IMPORTANCE_PATH` | `reports/feature_importance.csv` | Saved holdout permutation importance |
| `STUDENT_PERFORMANCE_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated explicit frontend origins |
| `STUDENT_PERFORMANCE_MAX_REQUEST_BYTES` | `16384` | Request size cap (1024–1048576 bytes) |

All relative artifact paths resolve from the project root. CORS origins are an
explicit comma-separated allow-list; wildcard origins are rejected and credentials
are disabled.

The model is loaded once at startup. `/api/v1/health` reports service availability
separately from model readiness; missing or incompatible artifacts leave the service
available but model-dependent routes return HTTP 503. Analytics are loaded from the
configured real CSV and are never fabricated. The distribution endpoint accepts an
observed dataset column name and returns bounded histogram bins or category counts.

Run API and project regression tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Verified in the current workspace: Python 3.14.7, FastAPI 0.162.0, Pydantic 2.14.1,
and Uvicorn 0.54.0. The saved pipeline generated the example prediction above; all
seven routes responded during a live local-server check. The full project test suite
reported 39 passed, with one Starlette deprecation warning for HTTPX-backed TestClient.
`compileall` and `pip check` also completed successfully. See the README for
verification details.

Do not load untrusted Joblib files. The model evaluation is from a single held-out
split and should not be interpreted as a guarantee of future performance. The source
and documented score range have not yet been independently verified; the observed
`Exam_Score` maximum of 101 is retained as supplied. MAE is low while RMSE is
substantially larger because the holdout residual tail includes a small number of
large errors. Feature importance and associations do not establish causation.
