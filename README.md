EDUPREDICT — STUDENT PERFORMANCE INTELLIGENCE
=============================================

EduPredict is a full-stack machine-learning project for exploring student-performance data and predicting examination scores. It combines a reproducible scikit-learn regression pipeline, a FastAPI backend, and a React/TypeScript web interface with interactive analytics and motion/3D visual elements.

The project is intended for learning, demonstration, and exploratory educational analysis. Predictions are estimates; they are not guaranteed outcomes and should not be used as the sole basis for consequential decisions about students.

FEATURES
--------

  - Dataset inspection and EDA: data-quality checks, descriptive statistics, distributions, relationship plots, and reports.
  - Leakage-aware ML workflow: preprocessing within scikit-learn pipelines, training-only cross-validation, model comparison, and a held-out test evaluation.
  - Regression models: Dummy Regressor, Linear Regression, Ridge Regression, Decision Tree, Random Forest, and Gradient Boosting candidates.
  - Model interpretation: permutation feature-importance results and diagnostic charts.
  - FastAPI backend: versioned endpoints for health, model information, data analytics, evaluation metrics, feature importance, and predictions.
  - React + TypeScript frontend: overview dashboard, analytics, model evaluation, insights, methodology, and prediction workflow.
  - Interactive UI: responsive layout and motion/3D effects using the frontend's configured animation and 3D libraries.
  - Validation and tests: input/schema validation and automated Python/frontend checks.

TECHNOLOGY STACK
----------------

Machine learning and backend

  - Python, Pandas, NumPy
  - scikit-learn, Joblib
  - Matplotlib, Seaborn
  - FastAPI, Pydantic, Uvicorn
  - Pytest

Frontend

  - React, TypeScript, Vite
  - Tailwind CSS
  - Framer Motion
  - Recharts
  - React Three Fiber / Three.js and related utilities, where configured in frontend/package.json

ARCHITECTURE
------------

flowchart TD
    A[Kaggle CSV] --> B[Validation and EDA]
    B --> C[Preprocessing Pipeline]
    C --> D[Model Training and Cross-Validation]
    D --> E[Holdout Evaluation]
    E --> F[Saved scikit-learn Pipeline]
    F --> G[FastAPI Backend]
    G <--> H[React / TypeScript Frontend]
    H --> I[Dashboard and Analytics]
    H --> J[Prediction Form]
    H --> K[Model Insights]

Important modules and artifacts
-------------------------------

  - src/student_performance/data/ — dataset loading, validation, and inspection.
  - src/student_performance/analysis/ — exploratory analysis and visualization generation.
  - src/student_performance/models/train.py — model comparison, tuning, evaluation, artifact serialization, and feature-importance reporting.
  - src/student_performance/models/predict.py — loading the saved pipeline and validating the raw feature schema.
  - src/student_performance/api/ — FastAPI routes, Pydantic contracts, settings, and service logic.
  - frontend/ — React/Vite application and frontend tests.
  - models/best_model.joblib — complete fitted preprocessing-and-model pipeline.
  - models/metadata.json — feature schema, dataset fingerprint, model configuration, environment versions, and measured results.
  - reports/ — data-quality summaries, model comparisons, final metrics, feature importance, and figures.

DATASET
-------

Dataset: Student Performance Factors on Kaggle (https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

The development dataset used in the recorded experiments contains 6,607 rows and 20 columns. The numerical target is Exam_Score; it is excluded from the 19 prediction input features.

The application expects the CSV at:

Dataset/StudentPerformanceFactors.csv

The raw CSV is excluded from Git by default. Download it from Kaggle, extract it, and place it at the path above before running data inspection, EDA, or model training. Check the dataset's licence and attribution terms before redistributing the data.

The dataset includes academic, behavioural, lifestyle, and educational-context attributes. Three categorical predictors had missing values during the recorded inspection. The training pipeline handles missing values through preprocessing rather than modifying the original CSV.

MODEL METHODOLOGY
-----------------

The project treats Exam_Score as a supervised regression target. The recorded workflow uses a reproducible train/test split, shuffled five-fold cross-validation on the training portion, and a separate holdout set for final evaluation. Preprocessing operations are placed inside scikit-learn pipelines so imputers, encoders, and scalers are fitted within the appropriate training partitions.

The reported model-selection metric is mean cross-validation MAE. The selected candidate in the recorded run was Ridge Regression with alpha=10.0.

Recorded evaluation results
---------------------------

These figures are from the reported training run; rerun the workflow to regenerate results if the dataset or training setup changes.

Metric  |  Ridge Regression — holdout set
MAE  |  0.44974
MSE  |  3.25093
RMSE  |  1.80303
R-squared  |  0.77001

The selected Ridge model's reported training-only cross-validation MAE was 0.50771 with a fold standard deviation of 0.03091.

For comparison, the median Dummy Regressor baseline recorded holdout MAE of 2.80787, RMSE of 3.77306, and R-squared of -0.00714.

MAE describes average absolute error; RMSE penalizes larger errors more strongly; R-squared measures explained variance relative to a baseline. R-squared is a regression metric, not classification accuracy.

The recorded holdout contained 1,322 observations. Seven absolute residuals exceeded 10 score points, and the maximum absolute residual was 27.9929. The difference between MAE and RMSE indicates that some errors are much larger than the typical absolute error. These cases warrant investigation; they should not be removed merely to improve metrics.

Limitations and responsible use
-------------------------------

  - One Exam_Score value of 101 was observed and retained unchanged. Its compatibility with the authoritative score scale and the dataset's provenance have not been independently verified.
  - The metrics describe one held-out split and may not generalize to different institutions or populations.
  - Correlation and permutation feature importance describe predictive associations; they do not prove that changing a feature will cause a change in performance.
  - The model should be used as an educational demonstration and analytical aid, not as the sole basis for consequential decisions about students.

SETUP
-----

Use the Python version and virtual environment configured for the project. The recorded development environment used Python 3.14.7, Node.js 24.20.0, and npm 11.19.0. Check the installed versions against the project's dependency files before changing the environment.

1. Install Python dependencies
------------------------------

Run these commands from the repository root in PowerShell:

py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Skip the virtual-environment creation command if .venv already exists. Install the Kaggle CSV as described in the Dataset section before running the data workflow.

2. Inspect the dataset and generate EDA
---------------------------------------

.\.venv\Scripts\python.exe src\inspect_data.py
.\.venv\Scripts\python.exe src\run_eda.py

Detailed reports are written under reports/; diagnostic logs are written under logs/. Charts are saved under reports/figures/. IQR-based outlier counts are review flags, not automatic removal instructions.

3. Train or regenerate the model
--------------------------------

.\.venv\Scripts\python.exe src\train_model.py

Training compares the supported candidate regressors, selects candidates using training-only cross-validation, evaluates the chosen model on the holdout set, and generates model artifacts and reports. Do not delete the currently working artifacts before verifying a new training run completed successfully.

Expected generated files include:

  - models/best_model.joblib
  - models/metadata.json
  - reports/model_comparison.csv
  - reports/final_metrics.json
  - reports/feature_importance.csv
  - reports/model_training_summary.txt
  - figures under reports/figures/

RUN THE APPLICATION LOCALLY
---------------------------

The application uses two local processes: the Python API and the React development server.

1. Start FastAPI
----------------

From the repository root, open the first terminal and run:

.\.venv\Scripts\python.exe -m uvicorn student_performance.api.main:app --app-dir src --host 127.0.0.1 --port 8000

Interactive API documentation is available at http://127.0.0.1:8000/docs (http://127.0.0.1:8000/docs). The OpenAPI schema is at http://127.0.0.1:8000/openapi.json (http://127.0.0.1:8000/openapi.json).

2. Configure and start the frontend
-----------------------------------

Open a second terminal:

cd frontend
npm ci
npm run dev

Open the local URL printed by Vite; the expected development origin is http://localhost:5173.

The frontend API base URL is configured through VITE_API_BASE_URL. The development fallback is http://127.0.0.1:8000. If you need to override it, create frontend/.env.local with:

VITE_API_BASE_URL=http://127.0.0.1:8000

Restart Vite after changing environment variables. The backend CORS configuration must include the exact frontend origin. Never use wildcard CORS origins for a deployment that permits credentials.

API ENDPOINTS
-------------

Method and route  |  Purpose
GET /api/v1/health  |  Service liveness and model readiness
GET /api/v1/model-info  |  Target, feature schema, model configuration, metadata, and evaluation results
GET /api/v1/analytics/summary  |  Dataset summary and missing-value counts
GET /api/v1/analytics/distributions?feature=Exam_Score  |  Bounded histogram or category-frequency data
GET /api/v1/models/metrics  |  Saved cross-validation comparison and holdout metrics
GET /api/v1/models/feature-importance  |  Saved permutation feature-importance results
POST /api/v1/predict  |  Predict from the saved pipeline using raw feature values

The prediction endpoint requires the 19 predictor fields, and Exam_Score must not be sent as an input. Numeric inputs must be finite; categorical values must be strings or null where supported. Unknown extra fields are rejected. The fitted encoder is configured to handle previously unseen categories without failing, but unfamiliar categories may still produce less reliable predictions.

See docs/inference_contract.md for the full input contract and operational notes.

TESTING AND QUALITY CHECKS
--------------------------

Run backend tests from the repository root:

.\.venv\Scripts\python.exe -m pytest -q

Run frontend checks from frontend/:

npm run lint
npm test
npm run build

The recorded backend verification included 39 passing tests. The frontend foundation milestone recorded passing lint, 16 frontend tests, and a successful production build. A later local integration check returned HTTP 200 OK for a real POST /api/v1/predict request. Rerun the checks after changes; these notes describe previously reported results and are not a claim that the current checkout has just been retested.

ENVIRONMENT CONFIGURATION
-------------------------

The backend supports these optional environment variables. Relative paths resolve from the project root:

Variable  |  Default
STUDENT_PERFORMANCE_MODEL_PATH  |  models/best_model.joblib
STUDENT_PERFORMANCE_METADATA_PATH  |  models/metadata.json
STUDENT_PERFORMANCE_DATASET_PATH  |  Dataset/StudentPerformanceFactors.csv
STUDENT_PERFORMANCE_MODEL_COMPARISON_PATH  |  reports/model_comparison.csv
STUDENT_PERFORMANCE_METRICS_PATH  |  reports/final_metrics.json
STUDENT_PERFORMANCE_FEATURE_IMPORTANCE_PATH  |  reports/feature_importance.csv
STUDENT_PERFORMANCE_CORS_ORIGINS  |  http://localhost:5173,http://127.0.0.1:5173
STUDENT_PERFORMANCE_MAX_REQUEST_BYTES  |  16384

No credentials are required for the local prototype. Do not commit real secrets, .env files, virtual environments, or node_modules/.

TROUBLESHOOTING
---------------

  - Model not ready: verify that models/best_model.joblib and models/metadata.json exist and belong to the same trusted training run. Check logs/backend.log for diagnostics.
  - Dataset or analytics unavailable: verify Dataset/StudentPerformanceFactors.csv exists and matches the inspected schema.
  - Metrics missing: regenerate reports by running the training script on the real dataset; never substitute example values.
  - Prediction returns HTTP 422: include all 19 expected predictor keys, omit Exam_Score, and use the correct JSON types.
  - Browser CORS error: ensure the exact frontend origin is present in STUDENT_PERFORMANCE_CORS_ORIGINS.
  - Frontend cannot reach the API: check that Uvicorn is running on the configured host/port and that VITE_API_BASE_URL matches it.
  - 3D content is unavailable: confirm WebGL support and use the application's fallback behaviour; core navigation and analytics should remain usable without the 3D scene.

FUTURE IMPROVEMENTS
-------------------

Possible extensions include stronger dataset-provenance verification, further robustness analysis, documented prediction uncertainty where methodologically justified, improved model monitoring, additional accessibility/performance testing, and carefully controlled deployment. External deployment and authentication are not claimed as completed in this README.

ACKNOWLEDGEMENT
---------------

The student-factors dataset is sourced from Kaggle. Credit and reuse are subject to the original dataset page's licence and terms.

EduPredict — Student Performance Intelligence
Machine learning · Educational data analysis · Full-stack web development