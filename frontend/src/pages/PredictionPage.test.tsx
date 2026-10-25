import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PredictionPage } from "./PredictionPage";
import { api } from "../services/api";
import {
  NUMERIC_PREDICTION_FIELDS,
  PREDICTION_FIELDS,
} from "../services/predictionForm";
import type {
  DistributionResponse,
  FeatureSchemaItem,
  ModelInfoResponse,
  PredictionResponse,
} from "../types/api";

vi.mock("../services/api", () => ({
  api: {
    modelInfo: vi.fn(),
    featureDistribution: vi.fn(),
    predict: vi.fn(),
  },
  ApiClientError: class ApiClientError extends Error {},
}));

const numericFields = new Set<string>(NUMERIC_PREDICTION_FIELDS);
const categoryValues: Record<string, string[]> = {
  Parental_Involvement: ["High", "Low", "Medium"],
  Access_to_Resources: ["High", "Low", "Medium"],
  Extracurricular_Activities: ["No", "Yes"],
  Motivation_Level: ["High", "Low", "Medium"],
  Internet_Access: ["No", "Yes"],
  Family_Income: ["High", "Low", "Medium"],
  Teacher_Quality: ["High", "Low", "Medium"],
  School_Type: ["Private", "Public"],
  Peer_Influence: ["Negative", "Neutral", "Positive"],
  Learning_Disabilities: ["No", "Yes"],
  Parental_Education_Level: ["College", "High School", "Postgraduate"],
  Distance_from_Home: ["Far", "Moderate", "Near"],
  Gender: ["Female", "Male"],
};

const schema: FeatureSchemaItem[] = PREDICTION_FIELDS.map((name) => ({
  name,
  data_type: numericFields.has(name) ? "number" : "string",
  nullable: true,
}));

const modelInfo: ModelInfoResponse = {
  target_name: "Exam_Score",
  selected_estimator: "Ridge",
  selected_candidate: "Ridge (tuned)",
  trained_feature_schema: schema,
  selected_hyperparameters: { alpha: 10 },
  training_timestamp_utc: "2026-10-24T17:24:29.116249+00:00",
  dataset_sha256: "dataset-fingerprint",
  cross_validation: {},
  split: {},
  library_versions: {},
  evaluation_results: {},
  limitations: ["Predictions are estimates."],
};

function distributionFor(feature: FeatureSchemaItem): DistributionResponse {
  if (feature.data_type === "number") {
    return {
      feature: feature.name,
      data_type: "int64",
      record_count: 6607,
      missing_count: 0,
      distribution_type: "histogram",
      bins: [{ start: 1, end: 44, count: 10 }],
      categories: [],
      truncated: false,
    };
  }

  return {
    feature: feature.name,
    data_type: "object",
    record_count: 6607,
    missing_count: 1,
    distribution_type: "category_frequency",
    bins: [],
    categories: [
      ...(categoryValues[feature.name] ?? []).map((category) => ({
        category,
        count: 10,
      })),
      { category: "__MISSING__", count: 1 },
    ],
    truncated: false,
  };
}

beforeEach(() => {
  vi.mocked(api.modelInfo).mockResolvedValue(modelInfo);
  vi.mocked(api.featureDistribution).mockImplementation(async (feature) => {
    const schemaItem = schema.find((item) => item.name === feature);
    if (!schemaItem) {
      throw new Error("Unknown test feature");
    }
    return distributionFor(schemaItem);
  });
  vi.mocked(api.predict).mockResolvedValue({
    target_name: "Exam_Score",
    predicted_score: 67.12516675552172,
    is_estimate: true,
    disclaimer: "This is an estimate for educational decision support.",
  });
});

async function renderReadyForm() {
  const user = userEvent.setup();
  render(<PredictionPage />);
  await screen.findByRole("heading", { name: "19 student factors" });
  return user;
}

describe("PredictionPage", () => {
  it("loads and renders the 19 nullable predictors without an Exam_Score input", async () => {
    const user = userEvent.setup();
    render(<PredictionPage />);
    expect(
      screen.getByText("Loading trained fields and observed category choices..."),
    ).toBeInTheDocument();
    await screen.findByRole("heading", { name: "19 student factors" });

    const form = screen.getByRole("button", { name: "Estimate examination score" }).closest("form");
    expect(form).not.toBeNull();
    expect(form?.querySelectorAll("input, select")).toHaveLength(19);
    expect(screen.queryByLabelText(/Exam_Score/i)).not.toBeInTheDocument();
    expect(screen.getByLabelText(/Hours studied/)).toHaveAttribute("type", "number");
    expect(screen.getByLabelText(/Hours studied/)).not.toHaveAttribute("min");
    expect(screen.getByLabelText(/Hours studied/)).not.toHaveAttribute("max");
    expect(screen.getByLabelText(/Parental involvement/).tagName).toBe("SELECT");
    expect(
      within(screen.getByLabelText(/Parental involvement/)).getByRole("option", {
        name: "High (10)",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText(/Bin edges are approximate guidance/).length,
    ).toBeGreaterThan(0);
    await user.click(screen.getByRole("button", { name: "Reset form" }));
  });

  it("submits all fields with real-category choices and renders the exact API result", async () => {
    const user = await renderReadyForm();

    await user.type(screen.getByLabelText(/Hours studied/), "23");
    await user.type(screen.getByLabelText(/Attendance/), "84");
    await user.selectOptions(screen.getByLabelText(/Parental involvement/), "Low");
    await user.selectOptions(screen.getByLabelText(/Access to resources/), "High");
    await user.selectOptions(
      screen.getByLabelText(/Extracurricular activities/),
      "No",
    );
    await user.click(screen.getByRole("button", { name: "Estimate examination score" }));

    await screen.findByText("67.12516675552172");
    expect(api.predict).toHaveBeenCalledTimes(1);
    const payload = vi.mocked(api.predict).mock.calls[0]?.[0];
    expect(payload).toMatchObject({
      Hours_Studied: 23,
      Attendance: 84,
      Parental_Involvement: "Low",
      Access_to_Resources: "High",
      Extracurricular_Activities: "No",
      Sleep_Hours: null,
      Previous_Scores: null,
    });
    expect(Object.keys(payload ?? {})).toHaveLength(19);
    expect(payload).not.toHaveProperty("Exam_Score");
    expect(await screen.findByText("Estimated examination score")).toBeInTheDocument();
    expect(screen.getAllByText(/not a guaranteed academic outcome/)).not.toHaveLength(0);
  });

  it("sends null for blank values and reports API failures with a retry action", async () => {
    const user = await renderReadyForm();
    vi.mocked(api.predict).mockRejectedValueOnce(
      new Error("The prediction service is unavailable."),
    );

    await user.click(screen.getByRole("button", { name: "Estimate examination score" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The prediction request could not be completed. Please try again.",
    );
    expect(api.predict).toHaveBeenCalledWith(
      Object.fromEntries(PREDICTION_FIELDS.map((field) => [field, null])),
      expect.any(AbortSignal),
    );

    await user.click(screen.getByRole("button", { name: "Try again" }));
    await screen.findByText("67.12516675552172");
  });

  it("resets entered values and clears a prior prediction", async () => {
    const user = await renderReadyForm();
    const hours = screen.getByLabelText(/Hours studied/);
    await user.type(hours, "23");
    await user.click(screen.getByRole("button", { name: "Estimate examination score" }));
    await screen.findByText("67.12516675552172");

    await user.click(screen.getByRole("button", { name: "Reset form" }));
    expect(hours).toHaveValue(null);
    expect(screen.queryByText("67.12516675552172")).not.toBeInTheDocument();
    expect(api.predict).toHaveBeenCalledTimes(1);
  });

  it("reports feature schema or distribution failures and allows retry", async () => {
    vi.mocked(api.modelInfo).mockRejectedValueOnce(new Error("offline"));
    render(<PredictionPage />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The prediction request could not be completed. Please try again.",
    );
    vi.mocked(api.modelInfo).mockResolvedValueOnce(modelInfo);
    await userEvent.setup().click(screen.getByRole("button", { name: "Try again" }));
    await screen.findByRole("heading", { name: "19 student factors" });
    expect(api.featureDistribution).toHaveBeenCalled();
  });

  it("does not submit invalid values or duplicate a pending request", async () => {
    const user = await renderReadyForm();
    const predictionRequest = new Promise<PredictionResponse>(() => undefined);
    vi.mocked(api.predict).mockReturnValueOnce(predictionRequest);
    const submit = screen.getByRole("button", { name: "Estimate examination score" });
    await user.click(submit);
    expect(await screen.findByRole("button", { name: "Sending to model..." })).toBeDisabled();
    expect(api.predict).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Reset form" }));
    await waitFor(() =>
      expect(screen.queryByText("Sending to model...")).not.toBeInTheDocument(),
    );
    expect(api.predict).toHaveBeenCalledTimes(1);
  });

  it("cancels an in-flight prediction when the form is reset", async () => {
    const user = await renderReadyForm();
    let capturedSignal: AbortSignal | undefined;
    vi.mocked(api.predict).mockImplementationOnce((_payload, signal) => {
      capturedSignal = signal;
      return new Promise(() => undefined);
    });

    await user.click(screen.getByRole("button", { name: "Estimate examination score" }));
    await screen.findByRole("button", { name: "Sending to model..." });
    await user.click(screen.getByRole("button", { name: "Reset form" }));
    expect(capturedSignal?.aborted).toBe(true);
    expect(screen.queryByText("Sending to model...")).not.toBeInTheDocument();
  });
});
