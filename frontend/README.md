# EduPredict frontend

The frontend is a standalone React/TypeScript application. It talks to the existing
FastAPI service; it does not duplicate preprocessing, model inference, or analytics.
It includes the responsive application shell, real-data dashboard and distribution
chart, saved evaluation/importance views, typed API client, a lazy-loaded interactive
Three.js landing scene, and a prediction form connected to the saved model.

## Requirements

- Node.js 20.19 or newer (verified with Node.js 24.20.0)
- npm (verified with npm 11.19.0)
- The existing Python virtual environment and backend dependencies for API features

Dependencies are pinned exactly in `package.json`. Keep `package-lock.json` in this
folder and install using `npm ci` for reproducible setup.

## Install and configure

From the project root:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env.local
```

`VITE_API_BASE_URL` selects the FastAPI origin. It defaults to
`http://127.0.0.1:8000`; do not include `/api/v1` in this value. The development
server uses `127.0.0.1:5173`, which is permitted by the backend's default CORS list.
Only put public frontend configuration in Vite variables; values are bundled into
client code.

## Run both applications

Start the Python API in one PowerShell terminal from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn student_performance.api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`. The API's interactive documentation is at
`http://127.0.0.1:8000/docs`. The header checks both `/api/v1/health` and
`/api/v1/model-info`; it reports backend availability separately from model
readiness. Dashboard, analytics, model, feature-importance, and methodology views
show API errors instead of substituting sample values.

## Routes

| Path | Page |
| --- | --- |
| `/` | Landing |
| `/dashboard` | Live dataset summary |
| `/analytics` | Backend-computed feature distribution |
| `/predict` | Prediction form using the model's trained feature schema |
| `/models` | Saved cross-validation and holdout evaluation |
| `/insights` | Saved permutation-importance results |
| `/about` | Model metadata and limitations |

## API client

`src/services/api.ts` centralizes calls for health, model info, analytics summary,
feature distributions, model metrics, feature importance, and prediction. Request and
response contracts live in `src/types/api.ts` and match the current API v1 schemas.
The prediction form loads the 19 fields from model metadata and category choices from
the configured dataset's feature distributions. Blank fields are sent as `null`;
`Exam_Score` is not an input. Numeric histogram spans are approximate guidance only,
not validation limits. Requests have a 10-second timeout, accept an optional abort
signal, surface the API's safe error envelope, and never log student input.

The endpoint paths and error envelope are owned by the Python service. If its
contracts change, update these types and their focused tests against the generated
OpenAPI schema.

## Development checks

```powershell
npm run lint
npm test
npm run build
npm run preview
```

Vitest runs in jsdom. Tests cover route rendering/navigation, backend-unavailable
status, API URL/error handling, reduced-motion transitions, prediction field and
payload contracts, validation, submission states, and the decorative scene fallback.
The landing page and shared shell render while the backend is stopped; data-dependent
pages then show an explicit unavailable state.

Verified in this workspace on Node.js 24.20.0 and npm 11.19.0:

- A clean `npm install --no-audit --no-fund` completed with the pinned dependencies.
- `npm run lint` passed.
- `npm test` passed: 28 tests across 6 files.
- `npm run build` passed TypeScript checking and produced the Vite production bundle.
- A local browser check rendered the landing page and 3D canvas, loaded the live
  prediction form with all 19 fields, and submitted a real prediction to the local
  FastAPI model. The form displayed the exact API result. The browser viewport had
  no horizontal overflow after the responsive-width adjustment.
- The browser reported a `THREE.Clock` deprecation warning from the installed
  React Three Fiber dependency. The scene remained rendered; no frontend exceptions
  were observed after the API became available.
- The 3D scene is code-split into a lazy chunk; Vite reports that its minified chunk
  exceeds 500 kB (about 924 kB / 247 kB gzip). It is not part of the initial page
  bundle.

## Design and limitations

Tailwind CSS 3 uses the PostCSS configuration in this folder and a single
`tailwind.config.cjs` theme. The design tokens and accessible global focus/reduced
motion styles are in `src/styles/index.css`. Page transitions and section reveals use
Framer Motion with the `useReducedMotion` preference. The landing scene uses React
Three Fiber, Three.js, and Drei, with reduced rendering on compact/coarse-pointer
devices and a static illustration fallback when WebGL is unavailable.

The target score scale and source provenance remain unconfirmed; the observed
dataset includes a target value above 100. Model performance and feature importance
are reported from existing saved artifacts and are not guarantees or causal claims.
