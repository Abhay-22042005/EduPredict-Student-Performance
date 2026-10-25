import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppRoutes } from "./router";
import { api } from "../services/api";

vi.mock("../services/api", () => ({
  api: {
    health: vi.fn(),
    modelInfo: vi.fn(),
    analyticsSummary: vi.fn(),
    featureDistribution: vi.fn(),
    modelMetrics: vi.fn(),
    featureImportance: vi.fn(),
    predict: vi.fn(),
  },
  ApiClientError: class ApiClientError extends Error {},
}));

vi.mock("../components/landing/DecorativeDataScene", () => ({
  DecorativeDataScene: () => <div data-testid="decorative-data-scene" />,
}));

vi.mock("recharts", async () => {
  const React = await import("react");
  const Container = ({ children }: { children?: ReactNode }) =>
    React.createElement("div", null, children);
  const Leaf = () => null;

  return {
    Bar: Leaf,
    BarChart: Container,
    CartesianGrid: Leaf,
    ResponsiveContainer: Container,
    Tooltip: Leaf,
    XAxis: Leaf,
    YAxis: Leaf,
  };
});

const modelInfo = {
  target_name: "Exam_Score",
  selected_estimator: "Ridge",
  selected_candidate: "Ridge (tuned)",
  trained_feature_schema: [
    { name: "Hours_Studied", data_type: "number", nullable: true },
    { name: "Attendance", data_type: "number", nullable: true },
  ],
  selected_hyperparameters: { alpha: 10 },
  training_timestamp_utc: "2026-10-24T17:24:29.116249+00:00",
  dataset_sha256: "abc123456789",
  cross_validation: {},
  split: {},
  library_versions: {},
  evaluation_results: {},
  limitations: ["Predictions are estimates."],
};

const summary = {
  record_count: 6607,
  column_count: 20,
  feature_count: 19,
  numerical_feature_count: 6,
  categorical_feature_count: 13,
  duplicate_record_count: 0,
  missing_value_counts: { Teacher_Quality: 78 },
  target_name: "Exam_Score",
  target: {
    count: 6607,
    missing_count: 0,
    mean: 67.24,
    median: 67,
    standard_deviation: 3.89,
    minimum: 55,
    maximum: 101,
  },
};

beforeEach(() => {
  vi.mocked(api.health).mockResolvedValue({
    status: "ok",
    service_available: true,
    model_ready: true,
  });
  vi.mocked(api.modelInfo).mockResolvedValue(modelInfo);
  vi.mocked(api.analyticsSummary).mockResolvedValue(summary);
  vi.mocked(api.featureDistribution).mockResolvedValue({
    feature: "Hours_Studied",
    data_type: "int64",
    record_count: 6607,
    missing_count: 0,
    distribution_type: "histogram",
    bins: [{ start: 1, end: 2, count: 10 }],
    categories: [],
    truncated: false,
  });
  vi.mocked(api.modelMetrics).mockResolvedValue({
    selection_method: "Cross-validation MAE on training data.",
    cross_validation_results: [
      {
        selection_rank: 1,
        candidate: "Ridge (tuned)",
        tuned: true,
        mean_cv_mae: 0.5,
        std_cv_mae: 0.03,
        parameters: { alpha: 10 },
      },
    ],
    holdout_test_results: {
      selected_model_test: {
        mae: 0.4497,
        mse: 3.2509,
        rmse: 1.803,
        r2: 0.77,
      },
    },
  });
  vi.mocked(api.featureImportance).mockResolvedValue({
    method: "Permutation importance scored by increase in MAE.",
    evaluation_data: "Untouched holdout set.",
    causal_claim: false,
    items: [],
  });
});

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AppRoutes />
    </MemoryRouter>,
  );
}

describe("application routes", () => {
  it.each([
    ["/", /Student Performance Intelligence/],
    ["/dashboard", /A clear view of the data/],
    ["/analytics", /Explore observed distributions/],
    ["/predict", /A model estimate, with context/],
    ["/models", /Evaluation, separated by purpose/],
    ["/insights", /Interpret the model, not the student/],
    ["/about", /Transparent by design/],
  ])("renders the shared shell and page for %s", async (path, heading) => {
    renderAt(path);
    expect(
      await screen.findByRole("heading", { name: heading }, { timeout: 15_000 }),
    ).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Main navigation" })).toBeInTheDocument();
    expect(await screen.findByLabelText(/Backend connected, model ready/)).toBeInTheDocument();
  }, 15_000);

  it("navigates between routed pages through the shared navigation", async () => {
    renderAt("/");
    fireEvent.click(screen.getByRole("link", { name: "Overview" }));
    expect(
      await screen.findByRole("heading", { name: /A clear view of the data/ }),
    ).toBeInTheDocument();
  });

  it("links from the landing page to the dashboard and prediction routes", async () => {
    renderAt("/");
    expect(screen.getByRole("link", { name: /Explore dashboard/ })).toHaveAttribute(
      "href",
      "/dashboard",
    );
    expect(screen.getByRole("link", { name: /Try a prediction/ })).toHaveAttribute(
      "href",
      "/predict",
    );
  });

  it("renders holdout metrics from the backend's saved report shape", async () => {
    renderAt("/models");
    expect(await screen.findByText("0.4497")).toBeInTheDocument();
    expect(screen.getByText("3.2509")).toBeInTheDocument();
    expect(screen.getByText("1.8030")).toBeInTheDocument();
    expect(screen.getByText("0.7700")).toBeInTheDocument();
  });

  it("shows honest unavailable status when the API cannot be reached", async () => {
    vi.mocked(api.health).mockRejectedValueOnce(new Error("offline"));
    vi.mocked(api.modelInfo).mockRejectedValueOnce(new Error("offline"));

    renderAt("/");
    expect(await screen.findByText("Backend unavailable")).toBeInTheDocument();
    await waitFor(() => expect(api.modelInfo).toHaveBeenCalled());
  });
});
