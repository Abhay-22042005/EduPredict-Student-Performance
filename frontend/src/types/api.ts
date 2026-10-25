export interface HealthResponse {
  status: string;
  service_available: boolean;
  model_ready: boolean;
}

export interface FeatureSchemaItem {
  name: string;
  data_type: string;
  nullable: boolean;
}

export interface ModelInfoResponse {
  target_name: string;
  selected_estimator: string;
  selected_candidate: string;
  trained_feature_schema: FeatureSchemaItem[];
  selected_hyperparameters: Record<string, unknown>;
  training_timestamp_utc: string;
  dataset_sha256: string;
  cross_validation: Record<string, unknown>;
  split: Record<string, unknown>;
  library_versions: Record<string, string>;
  evaluation_results: Record<string, unknown>;
  limitations: string[];
}

export interface TargetSummary {
  count: number;
  missing_count: number;
  mean: number;
  median: number;
  standard_deviation: number;
  minimum: number;
  maximum: number;
}

export interface AnalyticsSummaryResponse {
  record_count: number;
  column_count: number;
  feature_count: number;
  numerical_feature_count: number;
  categorical_feature_count: number;
  duplicate_record_count: number;
  missing_value_counts: Record<string, number>;
  target_name: string;
  target: TargetSummary;
}

export interface HistogramBin {
  start: number;
  end: number;
  count: number;
}

export interface CategoryCount {
  category: string;
  count: number;
}

export interface DistributionResponse {
  feature: string;
  data_type: string;
  record_count: number;
  missing_count: number;
  distribution_type: string;
  bins: HistogramBin[];
  categories: CategoryCount[];
  truncated: boolean;
}

export interface CrossValidationResult {
  selection_rank: number;
  candidate: string;
  tuned: boolean;
  mean_cv_mae: number;
  std_cv_mae: number;
  parameters: Record<string, unknown>;
}

export interface ModelMetricsResponse {
  selection_method: string;
  cross_validation_results: CrossValidationResult[];
  holdout_test_results: HoldoutTestResults;
}

export interface RegressionMetrics {
  mae: number;
  mse: number;
  rmse: number;
  r2: number;
  [metric: string]: number;
}

export interface HoldoutTestResults {
  selected_model_test?: RegressionMetrics;
  dummy_baseline_test?: RegressionMetrics;
  selected_model?: string;
  selected_cv_mae_mean?: number;
  selected_cv_mae_std?: number;
  selection_metric?: string;
  test_rows?: number;
  [key: string]: unknown;
}

export interface FeatureImportanceItem {
  feature: string;
  importance_mean: number;
  importance_std: number;
  importance_direction: string;
  interpretation: string;
}

export interface FeatureImportanceResponse {
  method: string;
  evaluation_data: string;
  causal_claim: boolean;
  items: FeatureImportanceItem[];
}

export interface PredictionRequest {
  Hours_Studied: number | null;
  Attendance: number | null;
  Parental_Involvement: string | null;
  Access_to_Resources: string | null;
  Extracurricular_Activities: string | null;
  Sleep_Hours: number | null;
  Previous_Scores: number | null;
  Motivation_Level: string | null;
  Internet_Access: string | null;
  Tutoring_Sessions: number | null;
  Family_Income: string | null;
  Teacher_Quality: string | null;
  School_Type: string | null;
  Peer_Influence: string | null;
  Physical_Activity: number | null;
  Learning_Disabilities: string | null;
  Parental_Education_Level: string | null;
  Distance_from_Home: string | null;
  Gender: string | null;
}

export interface PredictionResponse {
  target_name: string;
  predicted_score: number;
  is_estimate: boolean;
  disclaimer: string;
}

export interface ApiErrorResponse {
  error: {
    code: string;
    message: string;
    details?: Array<{
      field: string;
      issue: string;
    }>;
  };
}
