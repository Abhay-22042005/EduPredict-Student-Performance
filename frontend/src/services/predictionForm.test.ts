import { describe, expect, it } from "vitest";
import {
  buildPredictionPayload,
  NUMERIC_PREDICTION_FIELDS,
  PREDICTION_FIELDS,
  validatePredictionSchema,
} from "./predictionForm";
import type { FeatureSchemaItem, PredictionRequest } from "../types/api";

const numericFields = new Set<string>(NUMERIC_PREDICTION_FIELDS);
const schema: FeatureSchemaItem[] = PREDICTION_FIELDS.map((name) => ({
  name,
  data_type: numericFields.has(name) ? "number" : "string",
  nullable: true,
}));

const categories = {
  Parental_Involvement: ["Low", "Medium", "High"],
  Access_to_Resources: ["Low", "Medium", "High"],
  Extracurricular_Activities: ["No", "Yes"],
  Motivation_Level: ["Low", "Medium", "High"],
  Internet_Access: ["No", "Yes"],
  Family_Income: ["Low", "Medium", "High"],
  Teacher_Quality: ["Low", "Medium", "High"],
  School_Type: ["Private", "Public"],
  Peer_Influence: ["Negative", "Neutral", "Positive"],
  Learning_Disabilities: ["No", "Yes"],
  Parental_Education_Level: ["College", "High School", "Postgraduate"],
  Distance_from_Home: ["Far", "Moderate", "Near"],
  Gender: ["Female", "Male"],
} as const;

describe("prediction form contract", () => {
  it("accepts only the exact nullable 19-field trained schema", () => {
    expect(validatePredictionSchema(schema)).toBeNull();
    expect(validatePredictionSchema(schema.slice(1))).toContain("19-feature");
    expect(
      validatePredictionSchema(
        schema.map((field) =>
          field.name === "Attendance" ? { ...field, nullable: false } : field,
        ),
      ),
    ).toContain("19-feature");
    expect(validatePredictionSchema([...schema, {
      name: "Exam_Score",
      data_type: "number",
      nullable: true,
    }])).toContain("19-feature");
  });

  it("builds all required fields, parses numeric values, and leaves blanks null", () => {
    const result = buildPredictionPayload(
      {
        Hours_Studied: "23",
        Attendance: "84.5",
        Parental_Involvement: "Low",
      },
      categories,
    );

    expect(result.errors).toEqual({});
    expect(result.payload).toEqual({
      Hours_Studied: 23,
      Attendance: 84.5,
      Parental_Involvement: "Low",
      Access_to_Resources: null,
      Extracurricular_Activities: null,
      Sleep_Hours: null,
      Previous_Scores: null,
      Motivation_Level: null,
      Internet_Access: null,
      Tutoring_Sessions: null,
      Family_Income: null,
      Teacher_Quality: null,
      School_Type: null,
      Peer_Influence: null,
      Physical_Activity: null,
      Learning_Disabilities: null,
      Parental_Education_Level: null,
      Distance_from_Home: null,
      Gender: null,
    } satisfies PredictionRequest);
    expect(Object.keys(result.payload ?? {})).toHaveLength(19);
    expect(result.payload).not.toHaveProperty("Exam_Score");
  });

  it("rejects non-finite numbers and categories outside observed choices", () => {
    const result = buildPredictionPayload(
      {
        Hours_Studied: "Infinity",
        Parental_Involvement: "Unobserved",
      },
      categories,
    );

    expect(result.payload).toBeUndefined();
    expect(result.errors).toMatchObject({
      Hours_Studied: expect.stringContaining("finite"),
      Parental_Involvement: expect.stringContaining("observed dataset"),
    });
  });
});
