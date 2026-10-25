import { useEffect, useState } from "react";
import { Activity, Database, Rows3, Target } from "lucide-react";
import { Link } from "react-router-dom";
import { api, ApiClientError } from "../services/api";
import type { AnalyticsSummaryResponse } from "../types/api";
import { Card } from "../components/ui/Card";
import { EmptyState } from "../components/ui/EmptyState";
import { ErrorState } from "../components/ui/ErrorState";
import { LoadingState } from "../components/ui/LoadingState";

type SummaryState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; summary: AnalyticsSummaryResponse };

function messageFor(error: unknown): string {
  return error instanceof ApiClientError
    ? error.message
    : "Dataset summary could not be loaded.";
}

export function DashboardPage() {
  const [refreshKey, setRefreshKey] = useState(0);
  const [state, setState] = useState<SummaryState>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    void api
      .analyticsSummary(controller.signal)
      .then((summary) => {
        if (!controller.signal.aborted) {
          setState({ kind: "ready", summary });
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
        <p className="eyebrow">Workspace / Overview</p>
        <h1 className="page-title mt-3">A clear view of the data.</h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted">
          This overview is populated from the configured dataset and saved backend
          artifacts. Nothing is shown as a measured result until the API provides it.
        </p>
      </div>

      {state.kind === "loading" ? <LoadingState label="Loading dataset summary..." /> : null}
      {state.kind === "error" ? (
        <ErrorState
          message={state.message}
          onRetry={() => {
            setState({ kind: "loading" });
            setRefreshKey((key) => key + 1);
          }}
        />
      ) : null}
      {state.kind === "ready" ? (
        <SummaryOverview summary={state.summary} />
      ) : null}
    </div>
  );
}

function SummaryOverview({ summary }: { summary: AnalyticsSummaryResponse }) {
  const stats = [
    {
      icon: Rows3,
      label: "Dataset records",
      value: summary.record_count.toLocaleString(),
      note: "Observed rows",
    },
    {
      icon: Database,
      label: "Predictor features",
      value: summary.feature_count.toLocaleString(),
      note: `${summary.numerical_feature_count} numeric / ${summary.categorical_feature_count} categorical`,
    },
    {
      icon: Target,
      label: summary.target_name,
      value: `${summary.target.minimum} - ${summary.target.maximum}`,
      note: "Observed target range",
    },
    {
      icon: Activity,
      label: "Duplicate records",
      value: summary.duplicate_record_count.toLocaleString(),
      note: "Exact full-row duplicates",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {stats.map(({ icon: Icon, label, value, note }) => (
          <Card className="min-h-40" key={label}>
            <div className="flex items-center justify-between gap-3">
              <p className="text-xs font-medium text-muted">{label}</p>
              <Icon aria-hidden="true" className="h-4 w-4 text-teal" />
            </div>
            <p className="metric-value mt-5 text-2xl font-bold">{value}</p>
            <p className="mt-2 text-[11px] text-muted">{note}</p>
          </Card>
        ))}
      </div>
      <Card>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="eyebrow">Target snapshot</p>
            <h2 className="mt-2 text-lg font-semibold text-white">
              {summary.target_name} / observed dataset
            </h2>
          </div>
          <Link className="text-sm font-semibold text-cyan hover:text-white" to="/analytics">
            Explore analytics
          </Link>
        </div>
        <div className="mt-6 grid gap-3 sm:grid-cols-3">
          {[
            ["Mean", summary.target.mean],
            ["Median", summary.target.median],
            ["Standard deviation", summary.target.standard_deviation],
          ].map(([label, value]) => (
            <div className="rounded-xl border border-line bg-white/[0.025] p-4" key={label}>
              <p className="text-xs text-muted">{label}</p>
              <p className="metric-value mt-2 text-xl font-semibold">
                {typeof value === "number" ? value.toFixed(2) : value}
              </p>
            </div>
          ))}
        </div>
        {summary.target.maximum > 100 ? (
          <p className="mt-5 border-t border-line pt-4 text-xs leading-5 text-amber-100/75">
            The observed target maximum is above 100. The score scale and data provenance
            have not been independently confirmed.
          </p>
        ) : null}
      </Card>
      {Object.keys(summary.missing_value_counts).length === 0 ? (
        <EmptyState
          title="No missing values were reported"
          description="The API reports no missing values across the current dataset."
        />
      ) : (
        <Card>
          <p className="eyebrow">Data completeness</p>
          <h2 className="mt-2 text-lg font-semibold text-white">Missing values by feature</h2>
          <ul className="mt-4 divide-y divide-line">
            {Object.entries(summary.missing_value_counts).map(([feature, count]) => (
              <li className="flex items-center justify-between py-3 text-sm" key={feature}>
                <span className="text-slate-200">{feature}</span>
                <span className="font-mono text-muted">{count.toLocaleString()}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}
