import { useEffect, useState } from "react";
import { api, ApiClientError } from "../services/api";
import type { ModelMetricsResponse } from "../types/api";
import { Card } from "../components/ui/Card";
import { ErrorState } from "../components/ui/ErrorState";
import { LoadingState } from "../components/ui/LoadingState";

type DataState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; metrics: ModelMetricsResponse };

function messageFor(error: unknown): string {
  return error instanceof ApiClientError
    ? error.message
    : "Model evaluation reports could not be loaded.";
}

export function ModelLabPage() {
  const [refreshKey, setRefreshKey] = useState(0);
  const [state, setState] = useState<DataState>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    void api
      .modelMetrics(controller.signal)
      .then((metrics) => {
        if (!controller.signal.aborted) {
          setState({ kind: "ready", metrics });
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
        <p className="eyebrow">Workspace / Model lab</p>
        <h1 className="page-title mt-3">Evaluation, separated by purpose.</h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted">
          Training-only cross-validation informs selection; the separate holdout results
          are reported as final evaluation. These figures are loaded from saved backend
          reports.
        </p>
      </div>
      {state.kind === "loading" ? (
        <LoadingState label="Loading saved model evaluation..." />
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
      {state.kind === "ready" ? <MetricsView metrics={state.metrics} /> : null}
    </div>
  );
}

function MetricsView({ metrics }: { metrics: ModelMetricsResponse }) {
  const selectedModel = metrics.holdout_test_results.selected_model_test;
  const measured = ["mae", "mse", "rmse", "r2"] as const;

  return (
    <div className="space-y-6">
      <Card>
        <p className="eyebrow">Holdout test</p>
        <h2 className="mt-2 text-lg font-semibold text-white">Saved final evaluation</h2>
        {selectedModel ? (
          <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {measured
              .filter((key) => typeof selectedModel[key] === "number")
              .map((key) => (
                <div className="rounded-xl border border-line bg-white/[0.025] p-4" key={key}>
                  <p className="text-xs uppercase tracking-wider text-muted">{key}</p>
                  <p className="metric-value mt-2 text-xl font-semibold">
                    {selectedModel[key].toFixed(4)}
                  </p>
                </div>
              ))}
          </div>
        ) : (
          <p className="mt-4 text-sm text-muted">
            The saved report does not include metrics in the expected format.
          </p>
        )}
        <p className="mt-5 border-t border-line pt-4 text-xs leading-5 text-muted">
          {metrics.selection_method}
        </p>
      </Card>
      <Card className="overflow-hidden">
        <div className="mb-4">
          <p className="eyebrow">Cross-validation / training split only</p>
          <h2 className="mt-2 text-lg font-semibold text-white">Candidate comparison</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[38rem] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-line text-xs text-muted">
                <th className="py-3 pr-4 font-medium">Rank</th>
                <th className="py-3 pr-4 font-medium">Candidate</th>
                <th className="py-3 pr-4 font-medium">Mean CV MAE</th>
                <th className="py-3 pr-4 font-medium">Std. deviation</th>
                <th className="py-3 font-medium">Tuned</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line/80">
              {metrics.cross_validation_results.map((result) => (
                <tr className="text-slate-200" key={`${result.selection_rank}-${result.candidate}`}>
                  <td className="py-3 pr-4 font-mono text-muted">{result.selection_rank}</td>
                  <td className="py-3 pr-4 font-medium">{result.candidate}</td>
                  <td className="py-3 pr-4 font-mono">
                    {result.mean_cv_mae.toFixed(4)}
                  </td>
                  <td className="py-3 pr-4 font-mono">
                    {result.std_cv_mae.toFixed(4)}
                  </td>
                  <td className="py-3">{result.tuned ? "Yes" : "No"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
