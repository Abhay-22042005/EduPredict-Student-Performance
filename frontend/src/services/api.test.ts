import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiClientError } from "./api";
import type { PredictionRequest } from "../types/api";

describe("API client", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("requests the versioned health route and returns its response", async () => {
    const payload = {
      status: "ok",
      service_available: true,
      model_ready: true,
    };
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify(payload), { status: 200 }),
    );

    await expect(api.health()).resolves.toEqual(payload);
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/health",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("encodes feature names in the distribution query", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(
        JSON.stringify({
          feature: "Parental Education/Level",
          data_type: "object",
          record_count: 1,
          missing_count: 0,
          distribution_type: "category_frequency",
          bins: [],
          categories: [],
          truncated: false,
        }),
        { status: 200 },
      ),
    );

    await api.featureDistribution("Parental Education/Level");
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/analytics/distributions?feature=Parental%20Education%2FLevel",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("posts exactly the 19 raw prediction fields and excludes the target", async () => {
    const input: PredictionRequest = {
      Hours_Studied: 23,
      Attendance: 84,
      Parental_Involvement: "Low",
      Access_to_Resources: "High",
      Extracurricular_Activities: "No",
      Sleep_Hours: 7,
      Previous_Scores: 73,
      Motivation_Level: "Low",
      Internet_Access: "Yes",
      Tutoring_Sessions: 0,
      Family_Income: "Low",
      Teacher_Quality: "Medium",
      School_Type: "Public",
      Peer_Influence: "Positive",
      Physical_Activity: 3,
      Learning_Disabilities: "No",
      Parental_Education_Level: "High School",
      Distance_from_Home: "Near",
      Gender: "Male",
    };
    vi.mocked(fetch).mockResolvedValue(
      new Response(
        JSON.stringify({
          target_name: "Exam_Score",
          predicted_score: 67.125,
          is_estimate: true,
          disclaimer: "Estimate only.",
        }),
        { status: 200 },
      ),
    );

    await api.predict(input);

    const requestInit = vi.mocked(fetch).mock.calls[0]?.[1];
    const body = JSON.parse(String(requestInit?.body)) as Record<string, unknown>;
    expect(requestInit?.method).toBe("POST");
    expect(body).toEqual(input);
    expect(Object.keys(body)).toHaveLength(19);
    expect(body).not.toHaveProperty("Exam_Score");
  });

  it("surfaces the safe message from the backend error envelope", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(
        JSON.stringify({
          error: {
            code: "dataset_unavailable",
            message: "The configured dataset is unavailable.",
          },
        }),
        { status: 503 },
      ),
    );

    await expect(api.analyticsSummary()).rejects.toMatchObject({
      name: "ApiClientError",
      kind: "http",
      status: 503,
      code: "dataset_unavailable",
      message: "The configured dataset is unavailable.",
    });
  });

  it("distinguishes malformed JSON and network failures", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("{not-json", { status: 200 }),
    );
    await expect(api.health()).rejects.toMatchObject({
      kind: "parse",
    } satisfies Partial<ApiClientError>);

    vi.mocked(fetch).mockRejectedValue(new TypeError("socket detail"));
    await expect(api.health()).rejects.toMatchObject({
      kind: "network",
      message: "Could not reach the backend. Check that the API is running and try again.",
    });
  });
});
