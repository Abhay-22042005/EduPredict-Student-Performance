import type {
  AnalyticsSummaryResponse,
  ApiErrorResponse,
  DistributionResponse,
  FeatureImportanceResponse,
  HealthResponse,
  ModelInfoResponse,
  ModelMetricsResponse,
  PredictionRequest,
  PredictionResponse,
} from "../types/api";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";
const REQUEST_TIMEOUT_MS = 10_000;

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const API_BASE_URL = (configuredBaseUrl || DEFAULT_API_BASE_URL).replace(/\/+$/, "");

export type ApiErrorKind = "network" | "http" | "parse" | "timeout" | "cancelled";

export class ApiClientError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;
  readonly code?: string;

  constructor(
    kind: ApiErrorKind,
    message: string,
    options: { status?: number; code?: string; cause?: unknown } = {},
  ) {
    super(message, { cause: options.cause });
    this.name = "ApiClientError";
    this.kind = kind;
    this.status = options.status;
    this.code = options.code;
  }
}

function isApiErrorResponse(value: unknown): value is ApiErrorResponse {
  if (typeof value !== "object" || value === null || !("error" in value)) {
    return false;
  }
  const error = value.error;
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    typeof error.code === "string" &&
    "message" in error &&
    typeof error.message === "string"
  );
}

async function readResponseBody(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch (cause) {
    throw new ApiClientError(
      "parse",
      "The backend returned a response that could not be read.",
      { status: response.status, cause },
    );
  }
}

async function request<T>(
  path: string,
  options: { method?: "GET" | "POST"; body?: unknown; signal?: AbortSignal } = {},
): Promise<T> {
  const controller = new AbortController();
  const timeoutId = globalThis.setTimeout(
    () => controller.abort(new DOMException("Request timed out.", "TimeoutError")),
    REQUEST_TIMEOUT_MS,
  );
  const onAbort = () => controller.abort(options.signal?.reason);

  if (options.signal?.aborted) {
    onAbort();
  } else {
    options.signal?.addEventListener("abort", onAbort, { once: true });
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1${path}`, {
      method: options.method ?? "GET",
      headers: options.body === undefined ? undefined : { "Content-Type": "application/json" },
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      signal: controller.signal,
    });
    const payload = await readResponseBody(response);

    if (!response.ok) {
      const apiError = isApiErrorResponse(payload) ? payload.error : undefined;
      throw new ApiClientError(
        "http",
        apiError?.message ?? `The backend request failed with HTTP ${response.status}.`,
        {
          status: response.status,
          code: apiError?.code,
        },
      );
    }

    return payload as T;
  } catch (cause) {
    if (cause instanceof ApiClientError) {
      throw cause;
    }
    if (controller.signal.aborted) {
      const timedOut =
        controller.signal.reason instanceof DOMException &&
        controller.signal.reason.name === "TimeoutError";
      throw new ApiClientError(
        timedOut ? "timeout" : "cancelled",
        timedOut
          ? "The backend did not respond before the request timed out."
          : "The backend request was cancelled.",
        { cause },
      );
    }
    throw new ApiClientError(
      "network",
      "Could not reach the backend. Check that the API is running and try again.",
      { cause },
    );
  } finally {
    globalThis.clearTimeout(timeoutId);
    options.signal?.removeEventListener("abort", onAbort);
  }
}

export const api = {
  health: (signal?: AbortSignal) =>
    request<HealthResponse>("/health", { signal }),
  modelInfo: (signal?: AbortSignal) =>
    request<ModelInfoResponse>("/model-info", { signal }),
  analyticsSummary: (signal?: AbortSignal) =>
    request<AnalyticsSummaryResponse>("/analytics/summary", { signal }),
  featureDistribution: (feature: string, signal?: AbortSignal) =>
    request<DistributionResponse>(
      `/analytics/distributions?feature=${encodeURIComponent(feature)}`,
      { signal },
    ),
  modelMetrics: (signal?: AbortSignal) =>
    request<ModelMetricsResponse>("/models/metrics", { signal }),
  featureImportance: (signal?: AbortSignal) =>
    request<FeatureImportanceResponse>("/models/feature-importance", { signal }),
  predict: (input: PredictionRequest, signal?: AbortSignal) =>
    request<PredictionResponse>("/predict", {
      method: "POST",
      body: input,
      signal,
    }),
};

export { API_BASE_URL };
