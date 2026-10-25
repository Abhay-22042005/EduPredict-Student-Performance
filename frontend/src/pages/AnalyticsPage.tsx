import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiClientError } from "../services/api";
import type { DistributionResponse, ModelInfoResponse } from "../types/api";
import { Card } from "../components/ui/Card";
import { ErrorState } from "../components/ui/ErrorState";
import { LoadingState } from "../components/ui/LoadingState";

type FeatureState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; modelInfo: ModelInfoResponse };

type DistributionState =
  | { kind: "idle" | "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; data: DistributionResponse };

function messageFor(error: unknown): string {
  return error instanceof ApiClientError
    ? error.message
    : "The requested distribution could not be loaded.";
}

export function AnalyticsPage() {
  const [featureState, setFeatureState] = useState<FeatureState>({ kind: "loading" });
  const [distributionState, setDistributionState] = useState<DistributionState>({
    kind: "idle",
  });
  const [selectedFeature, setSelectedFeature] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    void api
      .modelInfo(controller.signal)
      .then((modelInfo) => {
        if (!controller.signal.aborted) {
          setFeatureState({ kind: "ready", modelInfo });
          setSelectedFeature((current) => current || modelInfo.trained_feature_schema[0]?.name || "");
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setFeatureState({ kind: "error", message: messageFor(error) });
        }
      });
    return () => controller.abort();
  }, [refreshKey]);

  useEffect(() => {
    if (!selectedFeature) {
      return;
    }
    const controller = new AbortController();
    void api
      .featureDistribution(selectedFeature, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) {
          setDistributionState({ kind: "ready", data });
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setDistributionState({ kind: "error", message: messageFor(error) });
        }
      });
    return () => controller.abort();
  }, [selectedFeature, refreshKey]);

  return (
    <div className="space-y-8">
      <div>
        <p className="eyebrow">Workspace / Analytics</p>
        <h1 className="page-title mt-3">Explore observed distributions.</h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted">
          Histograms and category frequencies are returned by the backend from the
          configured dataset. These summaries describe observations, not causes.
        </p>
      </div>

      {featureState.kind === "loading" ? (
        <LoadingState label="Loading the trained feature schema..." />
      ) : null}
      {featureState.kind === "error" ? (
        <ErrorState
          message={featureState.message}
          onRetry={() => {
            setFeatureState({ kind: "loading" });
            setDistributionState({ kind: "idle" });
            setRefreshKey((key) => key + 1);
          }}
        />
      ) : null}
      {featureState.kind === "ready" ? (
        <>
          <Card>
            <label className="block">
              <span className="text-xs font-medium text-muted">Observed feature</span>
              <select
                className="mt-2 block min-h-11 w-full rounded-xl border border-line bg-[#0b1727] px-3 text-sm text-slate-100 sm:max-w-md"
                onChange={(event) => {
                  setSelectedFeature(event.target.value);
                  setDistributionState({ kind: "loading" });
                }}
                value={selectedFeature}
              >
                {featureState.modelInfo.trained_feature_schema.map((feature) => (
                  <option key={feature.name} value={feature.name}>
                    {feature.name}
                  </option>
                ))}
              </select>
            </label>
          </Card>
          {distributionState.kind === "loading" || distributionState.kind === "idle" ? (
            <LoadingState label={`Loading ${selectedFeature} distribution...`} />
          ) : null}
          {distributionState.kind === "error" ? (
            <ErrorState
              message={distributionState.message}
              onRetry={() => {
                setDistributionState({ kind: "loading" });
                setRefreshKey((key) => key + 1);
              }}
            />
          ) : null}
          {distributionState.kind === "ready" ? (
            <DistributionChart data={distributionState.data} />
          ) : null}
        </>
      ) : null}
    </div>
  );
}

function DistributionChart({ data }: { data: DistributionResponse }) {
  const isNumeric = data.distribution_type === "histogram";
  const chartData = isNumeric
    ? data.bins.map((bin) => ({
        label: `${formatBound(bin.start)} - ${formatBound(bin.end)}`,
        count: bin.count,
      }))
    : data.categories.map((item) => ({
        label: item.category === "__MISSING__" ? "Missing" : item.category,
        count: item.count,
      }));

  return (
    <Card>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="eyebrow">Observed distribution</p>
          <h2 className="mt-2 text-lg font-semibold text-white">{data.feature}</h2>
          <p className="mt-1 text-xs text-muted">
            {data.record_count.toLocaleString()} records / {data.missing_count.toLocaleString()} missing
          </p>
        </div>
        <span className="rounded-lg border border-line px-2.5 py-1.5 font-mono text-[10px] text-slate-300">
          {isNumeric ? "HISTOGRAM" : "CATEGORY FREQUENCY"}
        </span>
      </div>
      <div className="mt-6 h-[18rem] w-full" role="img" aria-label={`${data.feature} distribution chart`}>
        <ResponsiveContainer
          height="100%"
          initialDimension={{ width: 320, height: 288 }}
          minWidth={0}
          width="100%"
        >
          <BarChart data={chartData} margin={{ top: 8, right: 10, bottom: 48, left: 0 }}>
            <CartesianGrid stroke="rgba(145, 163, 184, 0.12)" strokeDasharray="4 5" vertical={false} />
            <XAxis
              angle={-28}
              axisLine={false}
              dataKey="label"
              height={72}
              interval="preserveStartEnd"
              tick={{ fill: "#91a3b8", fontSize: 10 }}
              tickLine={false}
              textAnchor="end"
            />
            <YAxis
              allowDecimals={false}
              axisLine={false}
              tick={{ fill: "#91a3b8", fontSize: 10 }}
              tickLine={false}
              width={42}
            />
            <Tooltip
              contentStyle={{
                border: "1px solid rgba(139, 169, 196, 0.22)",
                borderRadius: 12,
                background: "#0b1727",
                color: "#e7eef6",
              }}
              cursor={{ fill: "rgba(105, 227, 227, 0.06)" }}
            />
            <Bar dataKey="count" fill="#39c6b1" maxBarSize={42} radius={[5, 5, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="border-t border-line pt-4 text-xs leading-5 text-muted">
        Counts are computed from the configured dataset. Distribution patterns alone do
        not establish relationships or causation.
      </p>
    </Card>
  );
}

function formatBound(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}
