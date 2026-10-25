import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { AlertCircle, Check, RotateCcw, ShieldCheck, Sparkles } from "lucide-react";
import { api, ApiClientError } from "../services/api";
import type {
  DistributionResponse,
  FeatureSchemaItem,
  ModelInfoResponse,
  PredictionRequest,
  PredictionResponse,
} from "../types/api";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { ErrorState } from "../components/ui/ErrorState";
import { LoadingState } from "../components/ui/LoadingState";
import { StatusBadge } from "../components/ui/StatusBadge";
import {
  buildPredictionPayload,
  getFieldLabel,
  groupPredictionFields,
  isNumericPredictionField,
  PREDICTION_FIELDS,
  validatePredictionSchema,
  type PredictionFormErrors,
  type PredictionFormValues,
} from "../services/predictionForm";

type SetupState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | {
      kind: "ready";
      modelInfo: ModelInfoResponse;
      distributions: Record<string, DistributionResponse>;
    };

type SubmissionState =
  | { kind: "idle" }
  | { kind: "submitting" }
  | { kind: "success"; response: PredictionResponse }
  | { kind: "error"; message: string; payload: PredictionRequest };

const PREDICTION_FIELD_SET = new Set<string>(PREDICTION_FIELDS);

function isPredictionField(name: string): name is keyof PredictionRequest {
  return PREDICTION_FIELD_SET.has(name);
}

function describeError(error: unknown): string {
  if (error instanceof ApiClientError) {
    return error.message;
  }
  return "The prediction request could not be completed. Please try again.";
}

function validateDistribution(
  feature: FeatureSchemaItem,
  distribution: DistributionResponse,
): void {
  const expectedType = feature.data_type === "number" ? "histogram" : "category_frequency";
  if (distribution.feature !== feature.name || distribution.distribution_type !== expectedType) {
    throw new Error(`The ${feature.name} distribution did not match its model schema.`);
  }
}

export function PredictionPage() {
  const [retryKey, setRetryKey] = useState(0);
  const [setup, setSetup] = useState<SetupState>({ kind: "loading" });
  const [values, setValues] = useState<PredictionFormValues>({});
  const [validationErrors, setValidationErrors] = useState<PredictionFormErrors>({});
  const [submission, setSubmission] = useState<SubmissionState>({ kind: "idle" });
  const requestController = useRef<AbortController | null>(null);
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    const controller = new AbortController();

    async function loadFormMetadata() {
      try {
        const modelInfo = await api.modelInfo(controller.signal);
        const schemaIssue = validatePredictionSchema(modelInfo.trained_feature_schema);
        if (schemaIssue) {
          throw new Error(schemaIssue);
        }

        const distributions = await Promise.all(
          modelInfo.trained_feature_schema.map(async (feature) => {
            const distribution = await api.featureDistribution(
              feature.name,
              controller.signal,
            );
            validateDistribution(feature, distribution);
            return [feature.name, distribution] as const;
          }),
        );

        if (!controller.signal.aborted) {
          setSetup({
            kind: "ready",
            modelInfo,
            distributions: Object.fromEntries(distributions),
          });
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          setSetup({ kind: "error", message: describeError(error) });
        }
      }
    }

    void loadFormMetadata();
    return () => controller.abort();
  }, [retryKey]);

  useEffect(
    () => () => {
      requestController.current?.abort();
    },
    [],
  );

  function updateValue(field: keyof PredictionRequest, value: string) {
    setValues((current) => ({ ...current, [field]: value }));
    setValidationErrors((current) => {
      if (!(field in current)) {
        return current;
      }
      const next = { ...current };
      delete next[field];
      return next;
    });
    setSubmission((current) => (current.kind === "idle" ? current : { kind: "idle" }));
  }

  function resetForm() {
    requestController.current?.abort();
    requestController.current = null;
    setValues({});
    setValidationErrors({});
    setSubmission({ kind: "idle" });
  }

  async function submitPayload(payload: PredictionRequest) {
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    setSubmission({ kind: "submitting" });

    try {
      const response = await api.predict(payload, controller.signal);
      if (!controller.signal.aborted) {
        setSubmission({ kind: "success", response });
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        setSubmission({ kind: "error", message: describeError(error), payload });
      }
    } finally {
      if (requestController.current === controller) {
        requestController.current = null;
      }
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (setup.kind !== "ready" || submission.kind === "submitting") {
      return;
    }

    const categoryOptions = Object.fromEntries(
      setup.modelInfo.trained_feature_schema
        .filter((feature) => feature.data_type === "string")
        .map((feature) => [
          feature.name,
          setup.distributions[feature.name]?.categories
            .filter((item) => item.category !== "__MISSING__")
            .map((item) => item.category) ?? [],
        ]),
    );
    const { payload, errors } = buildPredictionPayload(values, categoryOptions);
    setValidationErrors(errors);
    if (!payload) {
      setSubmission({ kind: "idle" });
      return;
    }
    void submitPayload(payload);
  }

  const groups =
    setup.kind === "ready"
      ? groupPredictionFields(setup.modelInfo.trained_feature_schema)
      : [];

  return (
    <div className="space-y-8">
      <div>
        <p className="eyebrow">Workspace / Prediction</p>
        <h1 className="page-title mt-3">A model estimate, with context.</h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted">
          Enter any observed student factors you have. Fields may be left blank; the
          saved model pipeline handles missing values. Every submitted request includes
          the full trained feature schema.
        </p>
      </div>

      <Card className="grid gap-5 sm:grid-cols-2">
        <div className="flex items-start gap-3">
          <ShieldCheck aria-hidden="true" className="mt-0.5 h-5 w-5 text-teal" />
          <div>
            <h2 className="text-sm font-semibold text-slate-100">Evidence, not a verdict</h2>
            <p className="mt-1 text-sm leading-6 text-muted">
              An estimated score is decision support, not a guaranteed academic outcome
              or definitive judgment.
            </p>
          </div>
        </div>
        <div className="flex items-start gap-3">
          <AlertCircle aria-hidden="true" className="mt-0.5 h-5 w-5 text-cyan" />
          <div>
            <h2 className="text-sm font-semibold text-slate-100">No score clipping</h2>
            <p className="mt-1 text-sm leading-6 text-muted">
              The result is displayed as returned by the saved model. The dataset's
              source score scale has not been independently confirmed.
            </p>
          </div>
        </div>
      </Card>

      {setup.kind === "loading" ? (
        <LoadingState label="Loading trained fields and observed category choices..." />
      ) : null}
      {setup.kind === "error" ? (
        <ErrorState
          message={setup.message}
          onRetry={() => {
            setSetup({ kind: "loading" });
            setRetryKey((key) => key + 1);
          }}
        />
      ) : null}

      {setup.kind === "ready" ? (
        <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_21rem]">
          <form className="space-y-5" noValidate onSubmit={handleSubmit}>
            <Card>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="eyebrow">Trained feature schema</p>
                  <h2 className="mt-2 text-lg font-semibold text-white">
                    {setup.modelInfo.trained_feature_schema.length} student factors
                  </h2>
                </div>
                <StatusBadge tone="ready">
                  {setup.modelInfo.selected_candidate}
                </StatusBadge>
              </div>
              <p className="mt-4 text-xs leading-5 text-muted">
                The API requires every feature key, but all 19 values are nullable.
                Leave a value blank to send null and use the model's fitted imputation.
                Category choices come from the configured dataset.
              </p>
            </Card>

            {groups.map((group, groupIndex) => (
              <motion.section
                className="space-y-3"
                key={group.title}
                initial={reduceMotion ? false : { opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: reduceMotion ? 0 : 0.28, delay: reduceMotion ? 0 : groupIndex * 0.045 }}
              >
                <div className="flex items-center justify-between gap-3 px-1">
                  <h2 className="text-sm font-semibold text-slate-100">{group.title}</h2>
                  <span className="text-[10px] text-muted">
                    {group.fields.length} {group.fields.length === 1 ? "factor" : "factors"}
                  </span>
                </div>
                <div className="grid gap-3 sm:grid-cols-2">
                  {group.fields.map((feature) => {
                    if (!isPredictionField(feature.name)) {
                      return null;
                    }
                    const field = feature.name;
                    const numeric = isNumericPredictionField(field);
                    const distribution = setup.distributions[field];
                    const selectedValue = values[field] ?? "";
                    const error = validationErrors[field];
                    const id = `prediction-${field}`;
                    const errorId = `${id}-error`;
                    const helpId = `${id}-help`;
                    const missingCategory = distribution?.categories.find(
                      (item) => item.category === "__MISSING__",
                    );
                    const firstBin = distribution?.bins[0];
                    const lastBin = distribution?.bins.at(-1);

                    return (
                      <div className="prediction-field" key={feature.name}>
                        <label
                          className="flex min-h-10 items-start justify-between gap-2 text-sm font-semibold text-slate-100"
                          htmlFor={id}
                        >
                          <span>{getFieldLabel(field)}</span>
                          <span className="shrink-0 text-[10px] font-normal text-muted">
                            Optional value
                          </span>
                        </label>

                        {numeric ? (
                          <input
                            aria-describedby={`${helpId}${error ? ` ${errorId}` : ""}`}
                            aria-invalid={Boolean(error)}
                            className={`prediction-control mt-2 ${error ? "border-rose-300/70" : ""}`}
                            disabled={submission.kind === "submitting"}
                            id={id}
                            inputMode="decimal"
                            onChange={(event) =>
                              updateValue(field, event.currentTarget.value)
                            }
                            placeholder="Enter a number or leave blank"
                            step="any"
                            type="number"
                            value={selectedValue}
                          />
                        ) : (
                          <select
                            aria-describedby={`${helpId}${error ? ` ${errorId}` : ""}`}
                            aria-invalid={Boolean(error)}
                            className={`prediction-control mt-2 ${error ? "border-rose-300/70" : ""}`}
                            disabled={submission.kind === "submitting"}
                            id={id}
                            onChange={(event) =>
                              updateValue(field, event.currentTarget.value)
                            }
                            value={selectedValue}
                          >
                            <option value="">Not provided (null)</option>
                            {distribution?.categories
                              .filter((item) => item.category !== "__MISSING__")
                              .map((item) => (
                                <option key={item.category} value={item.category}>
                                  {item.category} ({item.count.toLocaleString()})
                                </option>
                              ))}
                          </select>
                        )}

                        <p className="prediction-help mt-2" id={helpId}>
                          {numeric
                            ? firstBin && lastBin
                              ? `Observed histogram span: ${formatBound(firstBin.start)}-${formatBound(lastBin.end)}. Bin edges are approximate guidance, not input limits.`
                              : "Finite numeric value; no input range is enforced by the API."
                            : missingCategory
                              ? `${missingCategory.count.toLocaleString()} records are missing this value in the dataset.`
                              : "Choose a category observed in the configured dataset, or leave null."}
                        </p>
                        {error ? (
                          <p className="prediction-error mt-1" id={errorId} role="alert">
                            {error}
                          </p>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              </motion.section>
            ))}

            {submission.kind === "error" ? (
              <ErrorState
                message={submission.message}
                onRetry={() => void submitPayload(submission.payload)}
              />
            ) : null}

            <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-line bg-white/[0.02] p-4">
              <Button
                onClick={resetForm}
                type="button"
                variant="secondary"
              >
                <RotateCcw aria-hidden="true" className="h-4 w-4" />
                Reset form
              </Button>
              <Button
                className="min-w-48"
                disabled={submission.kind === "submitting"}
                type="submit"
                variant="primary"
              >
                {submission.kind === "submitting" ? (
                  <>
                    <span
                      aria-hidden="true"
                      className="h-4 w-4 animate-spin rounded-full border-2 border-cyan/30 border-t-cyan motion-reduce:animate-none"
                    />
                    Sending to model...
                  </>
                ) : (
                  <>
                    <Sparkles aria-hidden="true" className="h-4 w-4" />
                    Estimate examination score
                  </>
                )}
              </Button>
            </div>
          </form>

          <aside className="space-y-4 xl:sticky xl:top-24">
            {submission.kind === "success" ? (
              <PredictionResult response={submission.response} />
            ) : (
              <Card className="overflow-hidden">
                <div aria-hidden="true" className="soft-grid absolute inset-0 opacity-25" />
                <div className="relative">
                  <p className="eyebrow">Prediction result</p>
                  <h2 className="mt-2 text-lg font-semibold text-white">
                    Ready when you are.
                  </h2>
                  <p className="mt-3 text-sm leading-6 text-muted">
                    Submit the feature record to the saved model. The estimate will
                    appear here exactly as returned by the API.
                  </p>
                  <div className="mt-6 border-t border-line pt-4 text-xs leading-5 text-muted">
                    Model inputs never include Exam_Score. Blank values are sent as
                    null for the trained pipeline to impute.
                  </div>
                </div>
              </Card>
            )}
            <Card inset>
              <div className="flex items-start gap-3">
                <AlertCircle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-amber-200" />
                <p className="text-xs leading-5 text-muted">
                  Numeric histogram bounds are approximate. The backend exposes bin
                  edges, not exact observed minimum and maximum values; no values are
                  clipped or rejected based on those edges.
                </p>
              </div>
            </Card>
          </aside>
        </div>
      ) : null}
    </div>
  );
}

function PredictionResult({ response }: { response: PredictionResponse }) {
  const reduceMotion = useReducedMotion();

  return (
    <motion.section
      aria-live="polite"
      aria-atomic="true"
      className="surface-panel relative overflow-hidden rounded-2xl border-teal/20 p-5 sm:p-6"
      initial={reduceMotion ? false : { opacity: 0, y: 12, scale: 0.99 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: reduceMotion ? 0 : 0.32, ease: "easeOut" }}
      role="status"
    >
      <div
        aria-hidden="true"
        className="absolute -right-12 -top-12 h-40 w-40 rounded-full bg-teal/[0.07] blur-3xl"
      />
      <div className="relative">
        <StatusBadge tone="ready">
          <Check aria-hidden="true" className="h-3 w-3" />
          Model response
        </StatusBadge>
        <p className="eyebrow mt-6">Estimated {response.target_name}</p>
        <p className="prediction-score mt-2 break-all font-display text-5xl font-extrabold text-white sm:text-6xl">
          {String(response.predicted_score)}
        </p>
        <p className="mt-3 text-sm text-muted">Estimated examination score</p>
        <div className="mt-6 border-t border-line pt-4">
          <p className="text-xs leading-5 text-muted">
            {response.disclaimer}
          </p>
          <p className="mt-3 text-xs leading-5 text-amber-100/75">
            This estimate depends on the available dataset and is not a guaranteed
            academic outcome.
          </p>
        </div>
      </div>
    </motion.section>
  );
}

function formatBound(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}
