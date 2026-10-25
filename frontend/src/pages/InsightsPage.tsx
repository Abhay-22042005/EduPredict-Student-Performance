import { useEffect, useState } from "react";
import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import { api, ApiClientError } from "../services/api";
import type { FeatureImportanceResponse } from "../types/api";
import { Card } from "../components/ui/Card";
import { ErrorState } from "../components/ui/ErrorState";
import { LoadingState } from "../components/ui/LoadingState";

type DataState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; data: FeatureImportanceResponse };

function messageFor(error: unknown): string {
  return error instanceof ApiClientError
    ? error.message
    : "Saved feature-importance results could not be loaded.";
}

export function InsightsPage() {
  const [refreshKey, setRefreshKey] = useState(0);
  const [state, setState] = useState<DataState>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    void api
      .featureImportance(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) {
          setState({ kind: "ready", data });
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
        <p className="eyebrow">Workspace / Feature insights</p>
        <h1 className="page-title mt-3">Interpret the model, not the student.</h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted">
          Saved permutation results describe how model error changed when a feature
          was shuffled. They are not causal effects or a measure of a student's worth.
        </p>
      </div>
      {state.kind === "loading" ? (
        <LoadingState label="Loading saved feature-importance report..." />
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
      {state.kind === "ready" ? <ImportanceView data={state.data} /> : null}
    </div>
  );
}

function ImportanceView({ data }: { data: FeatureImportanceResponse }) {
  return (
    <div className="space-y-5">
      <Card>
        <p className="eyebrow">Method</p>
        <h2 className="mt-2 text-lg font-semibold text-white">{data.method}</h2>
        <div className="mt-4 flex flex-wrap gap-2 text-xs text-muted">
          <span className="rounded-lg border border-line px-3 py-2">
            Evaluation: {data.evaluation_data}
          </span>
          <span className="rounded-lg border border-line px-3 py-2">
            Causal claim: {data.causal_claim ? "Yes" : "No"}
          </span>
        </div>
      </Card>
      {data.items.map((item) => {
        const Icon =
          item.importance_direction === "increases_error_when_permuted"
            ? ArrowUp
            : item.importance_direction === "decreases_error_when_permuted"
              ? ArrowDown
              : Minus;
        return (
          <Card className="grid gap-4 sm:grid-cols-[minmax(12rem,0.7fr)_minmax(0,1.3fr)] sm:items-center" key={item.feature}>
            <div>
              <p className="text-sm font-semibold text-slate-100">{item.feature}</p>
              <p className="mt-1 text-xs text-muted">{item.importance_direction.replaceAll("_", " ")}</p>
            </div>
            <div className="flex items-start gap-3">
              <Icon aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-teal" />
              <div>
                <p className="font-mono text-sm text-slate-200">
                  {item.importance_mean.toFixed(5)} +/- {item.importance_std.toFixed(5)}
                </p>
                <p className="mt-1 text-xs leading-5 text-muted">{item.interpretation}</p>
              </div>
            </div>
          </Card>
        );
      })}
    </div>
  );
}
