import type { FeatureSchemaItem, PredictionRequest } from "../types/api";

export const PREDICTION_FIELDS = [
  "Hours_Studied",
  "Attendance",
  "Parental_Involvement",
  "Access_to_Resources",
  "Extracurricular_Activities",
  "Sleep_Hours",
  "Previous_Scores",
  "Motivation_Level",
  "Internet_Access",
  "Tutoring_Sessions",
  "Family_Income",
  "Teacher_Quality",
  "School_Type",
  "Peer_Influence",
  "Physical_Activity",
  "Learning_Disabilities",
  "Parental_Education_Level",
  "Distance_from_Home",
  "Gender",
] as const satisfies readonly (keyof PredictionRequest)[];

export const NUMERIC_PREDICTION_FIELDS = [
  "Hours_Studied",
  "Attendance",
  "Sleep_Hours",
  "Previous_Scores",
  "Tutoring_Sessions",
  "Physical_Activity",
] as const satisfies readonly (keyof PredictionRequest)[];

type NumericFeature = (typeof NUMERIC_PREDICTION_FIELDS)[number];
type CategoricalFeature = Exclude<keyof PredictionRequest, NumericFeature>;

export type PredictionFormValues = Partial<Record<keyof PredictionRequest, string>>;
export type PredictionFormErrors = Partial<Record<keyof PredictionRequest, string>>;

const FEATURE_GROUPS: Record<string, string> = {
  Hours_Studied: "Academic history",
  Attendance: "Academic history",
  Previous_Scores: "Academic history",
  Tutoring_Sessions: "Academic history",
  Sleep_Hours: "Habits and wellbeing",
  Physical_Activity: "Habits and wellbeing",
  Motivation_Level: "Habits and wellbeing",
  Extracurricular_Activities: "Habits and wellbeing",
  Parental_Involvement: "Learning environment",
  Access_to_Resources: "Learning environment",
  Internet_Access: "Learning environment",
  Teacher_Quality: "Learning environment",
  School_Type: "Learning environment",
  Peer_Influence: "Learning environment",
  Family_Income: "Learning environment",
  Learning_Disabilities: "Learning environment",
  Parental_Education_Level: "Learning environment",
  Distance_from_Home: "Learning environment",
  Gender: "Learning environment",
};

const FIELD_LABELS: Record<keyof PredictionRequest, string> = {
  Hours_Studied: "Hours studied",
  Attendance: "Attendance",
  Parental_Involvement: "Parental involvement",
  Access_to_Resources: "Access to resources",
  Extracurricular_Activities: "Extracurricular activities",
  Sleep_Hours: "Sleep hours",
  Previous_Scores: "Previous scores",
  Motivation_Level: "Motivation level",
  Internet_Access: "Internet access",
  Tutoring_Sessions: "Tutoring sessions",
  Family_Income: "Family income",
  Teacher_Quality: "Teacher quality",
  School_Type: "School type",
  Peer_Influence: "Peer influence",
  Physical_Activity: "Physical activity",
  Learning_Disabilities: "Learning disabilities",
  Parental_Education_Level: "Parental education level",
  Distance_from_Home: "Distance from home",
  Gender: "Gender",
};

export interface PredictionFieldGroup {
  title: string;
  fields: FeatureSchemaItem[];
}

export function getFieldLabel(field: keyof PredictionRequest): string {
  return FIELD_LABELS[field];
}

export function groupPredictionFields(
  schema: FeatureSchemaItem[],
): PredictionFieldGroup[] {
  const groups = new Map<string, FeatureSchemaItem[]>();
  for (const feature of schema) {
    const group = FEATURE_GROUPS[feature.name] ?? "Other trained features";
    const fields = groups.get(group) ?? [];
    fields.push(feature);
    groups.set(group, fields);
  }

  return [...groups].map(([title, fields]) => ({ title, fields }));
}

export function isNumericPredictionField(
  field: keyof PredictionRequest,
): field is NumericFeature {
  return (NUMERIC_PREDICTION_FIELDS as readonly string[]).includes(field);
}

export function validatePredictionSchema(schema: FeatureSchemaItem[]): string | null {
  if (
    schema.length !== PREDICTION_FIELDS.length ||
    schema.some((feature, index) => feature.name !== PREDICTION_FIELDS[index]) ||
    schema.some((feature) => !feature.nullable) ||
    schema.some((feature) => !["number", "string"].includes(feature.data_type)) ||
    schema.some((feature) => feature.name === "Exam_Score")
  ) {
    return "The trained model schema does not match the expected 19-feature prediction contract.";
  }
  return null;
}

export function buildPredictionPayload(
  values: PredictionFormValues,
  categoryOptions: Partial<Record<CategoricalFeature, readonly string[]>>,
): { payload?: PredictionRequest; errors: PredictionFormErrors } {
  const errors: PredictionFormErrors = {};

  const numberValue = (field: NumericFeature): number | null => {
    const rawValue = values[field]?.trim();
    if (!rawValue) {
      return null;
    }
    const numericValue = Number(rawValue);
    if (!Number.isFinite(numericValue)) {
      errors[field] = "Enter a finite number, or leave this value blank.";
      return null;
    }
    return numericValue;
  };

  const categoryValue = (field: CategoricalFeature): string | null => {
    const value = values[field]?.trim();
    if (!value) {
      return null;
    }
    const options = categoryOptions[field];
    if (!options?.includes(value)) {
      errors[field] = "Choose a value from the observed dataset categories.";
      return null;
    }
    return value;
  };

  const payload: PredictionRequest = {
    Hours_Studied: numberValue("Hours_Studied"),
    Attendance: numberValue("Attendance"),
    Parental_Involvement: categoryValue("Parental_Involvement"),
    Access_to_Resources: categoryValue("Access_to_Resources"),
    Extracurricular_Activities: categoryValue("Extracurricular_Activities"),
    Sleep_Hours: numberValue("Sleep_Hours"),
    Previous_Scores: numberValue("Previous_Scores"),
    Motivation_Level: categoryValue("Motivation_Level"),
    Internet_Access: categoryValue("Internet_Access"),
    Tutoring_Sessions: numberValue("Tutoring_Sessions"),
    Family_Income: categoryValue("Family_Income"),
    Teacher_Quality: categoryValue("Teacher_Quality"),
    School_Type: categoryValue("School_Type"),
    Peer_Influence: categoryValue("Peer_Influence"),
    Physical_Activity: numberValue("Physical_Activity"),
    Learning_Disabilities: categoryValue("Learning_Disabilities"),
    Parental_Education_Level: categoryValue("Parental_Education_Level"),
    Distance_from_Home: categoryValue("Distance_from_Home"),
    Gender: categoryValue("Gender"),
  };

  return Object.keys(errors).length > 0 ? { errors } : { payload, errors };
}
