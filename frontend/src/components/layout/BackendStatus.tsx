import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { api, ApiClientError } from "../../services/api";
import type { HealthResponse, ModelInfoResponse } from "../../types/api";
import { Button } from "../ui/Button";
import { StatusBadge } from "../ui/StatusBadge";

type StatusState =
  | { kind: "loading" }
  | { kind: "unavailable"; message: string }
  | {
      kind: "available";
      health: HealthResponse;
      modelInfo?: ModelInfoResponse;
      infoMessage?: string;
    };

function errorMessage(error: unknown): string {
  return error instanceof ApiClientError
    ? error.message
    : "The backend status could not be checked.";
}

export function BackendStatus() {
  const [refreshKey, setRefreshKey] = useState(0);
  const [state, setState] = useState<StatusState>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    void Promise.allSettled([
      api.health(controller.signal),
      api.modelInfo(controller.signal),
    ]).then(([healthResult, modelResult]) => {
      if (controller.signal.aborted) {
        return;
      }
      if (healthResult.status === "rejected") {
        setState({ kind: "unavailable", message: errorMessage(healthResult.reason) });
        return;
      }

      setState({
        kind: "available",
        health: healthResult.value,
        ...(modelResult.status === "fulfilled"
          ? { modelInfo: modelResult.value }
          : { infoMessage: errorMessage(modelResult.reason) }),
      });
    });

    return () => controller.abort();
  }, [refreshKey]);

  if (state.kind === "loading") {
    return (
      <div aria-label="Checking backend and model status" className="hidden sm:block" role="status">
        <StatusBadge tone="neutral">Checking service</StatusBadge>
      </div>
    );
  }

  if (state.kind === "unavailable") {
    return (
      <div className="flex items-center gap-2" title={state.message}>
        <StatusBadge tone="error">Backend unavailable</StatusBadge>
        <Button
          aria-label="Refresh backend status"
          className="h-9 w-9 px-0"
          onClick={() => {
            setState({ kind: "loading" });
            setRefreshKey((key) => key + 1);
          }}
          type="button"
        >
          <RefreshCw aria-hidden="true" className="h-3.5 w-3.5" />
        </Button>
      </div>
    );
  }

  const connected = state.health.service_available;
  const modelReady = state.health.model_ready;
  const badgeText = !connected
    ? "Service degraded"
    : modelReady
      ? "Model ready"
      : "Model unavailable";
  const tone = !connected ? "warning" : modelReady ? "ready" : "warning";
  const accessibleLabel = [
    connected ? "Backend connected" : "Backend degraded",
    modelReady ? "model ready" : "model not ready",
  ].join(", ");

  return (
    <div
      aria-label={accessibleLabel}
      className="flex items-center gap-2"
      title={
        state.infoMessage ??
        (state.modelInfo
          ? `${state.modelInfo.selected_estimator} model`
          : "Model information is not available.")
      }
    >
      <span className="hidden md:inline">
        <StatusBadge tone={tone}>{badgeText}</StatusBadge>
      </span>
      <span className="md:hidden">
        <StatusBadge tone={connected && modelReady ? "ready" : "warning"}>
          {connected && modelReady ? "Ready" : "Offline"}
        </StatusBadge>
      </span>
      <Button
        aria-label="Refresh backend status"
        className="h-9 w-9 px-0"
        onClick={() => {
          setState({ kind: "loading" });
          setRefreshKey((key) => key + 1);
        }}
        type="button"
      >
        <RefreshCw aria-hidden="true" className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}
