import { useEffect, useState } from "react";
import { BookOpen, CircleAlert, Database, ShieldCheck } from "lucide-react";
import { api, ApiClientError } from "../services/api";
import type { ModelInfoResponse } from "../types/api";
import { Card } from "../components/ui/Card";
import { ErrorState } from "../components/ui/ErrorState";
import { LoadingState } from "../components/ui/LoadingState";

type DataState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; modelInfo: ModelInfoResponse };

function messageFor(error: unknown): string {
  return error instanceof ApiClientError
    ? error.message
    : "Model methodology could not be loaded.";
}

export function AboutPage() {
  const [refreshKey, setRefreshKey] = useState(0);
  const [state, setState] = useState<DataState>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    void api
      .modelInfo(controller.signal)
      .then((modelInfo) => {
        if (!controller.signal.aborted) {
          setState({ kind: "ready", modelInfo });
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setState({ kind: "error", message: messageFor(error) });
        }
      });
    return () => controller.abort();
  }, [refreshKey]);

  return (
    <div className="space-y-8">
      <div>
        <p className="eyebrow">Workspace / Methodology</p>
        <h1 className="page-title mt-3">Transparent by design.</h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted">
          The model and data stay behind a local API. This page reports available
          training metadata and makes known limitations visible.
        </p>
      </div>
      {state.kind === "loading" ? (
        <LoadingState label="Loading model metadata..." />
      ) : null}
      {state.kind === "error" ? (
        <ErrorState
          message={state.message}
          onRetry={() => {
            setState({ kind: "loading" });
            setRefreshKey((key) => key + 1);
          }}
        />
      ) : null}
      {state.kind === "ready" ? <Methodology modelInfo={state.modelInfo} /> : null}
    </div>
  );
}

function Methodology({ modelInfo }: { modelInfo: ModelInfoResponse }) {
  return (
    <div className="space-y-5">
      <Card>
        <div className="flex items-start gap-4">
          <span className="brand-mark shrink-0">
            <BookOpen aria-hidden="true" className="h-5 w-5" />
          </span>
          <div>
            <p className="eyebrow">Selected estimator</p>
            <h2 className="mt-2 text-xl font-semibold text-white">
              {modelInfo.selected_candidate}
            </h2>
            <p className="mt-2 text-sm text-muted">
              Target: <span className="text-slate-200">{modelInfo.target_name}</span>
              <span className="mx-2 text-slate-600">/</span>
              {modelInfo.trained_feature_schema.length} trained predictors
            </p>
          </div>
        </div>
        <div className="mt-6 grid gap-3 sm:grid-cols-2">
          <MetadataLine
            icon={Database}
            label="Dataset fingerprint"
            value={`${modelInfo.dataset_sha256.slice(0, 12)}...`}
          />
          <MetadataLine
            icon={ShieldCheck}
            label="Training timestamp (UTC)"
            value={modelInfo.training_timestamp_utc}
          />
        </div>
      </Card>
      <Card>
        <div className="flex items-center gap-2">
          <CircleAlert aria-hidden="true" className="h-4 w-4 text-amber-200" />
          <h2 className="text-sm font-semibold text-slate-100">Known limitations</h2>
        </div>
        <ul className="mt-4 space-y-3">
          {modelInfo.limitations.map((limitation) => (
            <li className="flex gap-3 text-sm leading-6 text-muted" key={limitation}>
              <span aria-hidden="true" className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-200/80" />
              {limitation}
            </li>
          ))}
        </ul>
      </Card>
      <p className="px-1 text-xs leading-5 text-muted">
        Correlation and feature-importance summaries do not establish causation.
        Confirm data provenance and target scale before making substantive decisions.
      </p>
    </div>
  );
}

function MetadataLine({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Database;
  label: string;
  value: string;
}) {
  return (
    <div className="flex min-w-0 items-start gap-3 rounded-xl border border-line bg-white/[0.025] p-4">
      <Icon aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-teal" />
      <div className="min-w-0">
        <p className="text-xs text-muted">{label}</p>
        <p className="mt-1 break-all text-xs text-slate-200">{value}</p>
      </div>
    </div>
  );
}
