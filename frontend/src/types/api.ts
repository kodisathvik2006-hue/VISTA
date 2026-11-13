export type HealthResponse = {
  status: string;
  service: string;
  version: string;
  opencv: boolean;
  ai_tracking?: boolean;
  device?: string;
};

export type VideoMetadata = {
  duration_seconds: number;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
};

export type VideoAnalyzeResponse = {
  id: string;
  original_filename: string;
  stored_filename: string;
  size_bytes: number;
  metadata: VideoMetadata;
  preview_url: string;
};

export type ApiErrorBody = {
  code: string;
  message: string;
  details?: unknown;
};

export type ApiErrorResponse = {
  error: ApiErrorBody;
};

export class NeurovisionApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "NeurovisionApiError";
    this.status = status;
    this.code = code;
  }
}

export const VistaApiError = NeurovisionApiError;
export type VistaApiError = NeurovisionApiError;

// Phase 2 Types
export type TrackedDetection = {
  track_id: number | null;
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
};

export type FrameDetectionRecord = {
  frame_index: number;
  timestamp_seconds: number;
  detections: TrackedDetection[];
};

export type JobSummary = {
  total_frames_processed: number;
  total_detections: number;
  unique_tracks: number;
  class_counts: Record<string, number>;
  peak_objects_in_frame: number;
  processing_time_seconds: number;
  fps_processed: number;
};

export type JobResponse = {
  job_id: string;
  video_id: string;
  status: "queued" | "processing" | "completed" | "failed" | "cancelled";
  progress: number;
  current_frame: number;
  total_frames: number;
  error?: string | null;
  created_at: string;
  completed_at?: string | null;
  summary?: JobSummary | null;
  annotated_video_url?: string | null;
};

export type JobDetailResponse = JobResponse & {
  frames: FrameDetectionRecord[];
};

export type JobCreateRequest = {
  video_id: string;
  confidence_threshold?: number;
  classes?: string[];
  save_annotated_video?: boolean;
  max_frames?: number;
};

export type ModelClassesResponse = {
  count: number;
  classes: Record<string, string>;
  default_tracking_classes: string[];
};

// =========================================================================
// Investigation Platform Types (Investigation-Based Video Intelligence)
// =========================================================================

export type InvestigationMode =
  | "suspicious_activity"
  | "specific_target"
  | "accident_detection"
  | "altercation_detection";

export type ZonePolygon = {
  zone_id: string;
  name: string;
  points: [number, number][]; // normalized coordinates [0..1]
  is_restricted?: boolean;
};

export type SuspiciousActivityConfig = {
  restricted_zones?: ZonePolygon[];
  stop_duration_threshold_seconds?: number;
  repeated_crossing_threshold?: number;
  confidence_threshold?: number;
  classes?: string[];
};

export type TargetSelectionCriteria = {
  candidate_id?: number | null;
  target_class?: string | null;
  target_color?: string | null;
  target_plate?: string | null;
  seed_bbox?: [number, number, number, number] | null;
  seed_frame_index?: number;
};

export type SpecificTargetConfig = {
  target_criteria: TargetSelectionCriteria;
  confidence_threshold?: number;
  camera_zone_label?: string;
};

export type AccidentDetectionConfig = {
  confidence_threshold?: number;
  ttc_threshold_seconds?: number;
  min_speed_delta_ratio?: number;
};

export type AltercationDetectionConfig = {
  confidence_threshold?: number;
  proximity_threshold_pixels?: number;
  motion_energy_threshold?: number;
  target_model?: string;
};

export type InvestigationCreateRequest = {
  video_id: string;
  mode: InvestigationMode;
  suspicious_config?: SuspiciousActivityConfig;
  target_config?: SpecificTargetConfig;
  accident_config?: AccidentDetectionConfig;
  altercation_config?: AltercationDetectionConfig;
  save_annotated_video?: boolean;
  max_frames?: number;
};

export type CandidateDetection = {
  candidate_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  dominant_color: string;
};

export type CandidatesResponse = {
  video_id: string;
  frame_index: number;
  timestamp_seconds: number;
  candidates: CandidateDetection[];
  preview_url: string;
};

export type ReviewStatus =
  | "Pending Human Review"
  | "Verified Incident"
  | "False Positive - Dismissed";

export type InvestigationEvent = {
  event_id: string;
  event_type: string;
  label: string;
  description: string;
  start_time: number;
  end_time: number;
  frame_index: number;
  confidence: number;
  uncertainty_note: string;
  review_status: ReviewStatus;
  evidence_frame_url?: string | null;
  clip_url?: string | null;
  related_track_id?: number | null;
  metadata?: Record<string, unknown>;
};

export type StoppingInterval = {
  start: number;
  end: number;
  duration: number;
};

export type TargetTelemetry = {
  first_seen_timestamp?: number | null;
  last_seen_timestamp?: number | null;
  total_visible_duration_seconds: number;
  observed_stopping_intervals: StoppingInterval[];
  zones_entered: string[];
  average_confidence: number;
  lost_visibility_intervals: StoppingInterval[];
  assigned_track_id?: number | null;
  target_class?: string | null;
  target_color?: string | null;
  calibration_note?: string;
};

export type InvestigationSummary = {
  mode: InvestigationMode;
  total_events: number;
  events_by_type: Record<string, number>;
  review_pending_count: number;
  target_telemetry?: TargetTelemetry | null;
  processing_time_seconds: number;
  total_frames_processed: number;
  fps_processed: number;
  notes: string[];
};

export type InvestigationJobResponse = {
  job_id: string;
  video_id: string;
  mode: InvestigationMode;
  status: "queued" | "processing" | "completed" | "failed" | "cancelled";
  progress: number;
  current_frame: number;
  total_frames: number;
  error?: string | null;
  created_at: string;
  completed_at?: string | null;
  summary?: InvestigationSummary | null;
  annotated_video_url?: string | null;
};

export type InvestigationJobDetailResponse = InvestigationJobResponse & {
  events: InvestigationEvent[];
  evidence_frame_urls: string[];
};
