import type {
  CandidatesResponse,
  HealthResponse,
  InvestigationCreateRequest,
  InvestigationEvent,
  InvestigationJobDetailResponse,
  InvestigationJobResponse,
  JobCreateRequest,
  JobDetailResponse,
  JobResponse,
  ModelClassesResponse,
  ReviewStatus,
  VideoAnalyzeResponse,
} from "../types/api";
import { NeurovisionApiError, type ApiErrorResponse } from "../types/api";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

async function parseError(response: Response): Promise<NeurovisionApiError> {
  try {
    const body = (await response.json()) as ApiErrorResponse;
    if (body?.error?.message) {
      return new NeurovisionApiError(response.status, body.error.code, body.error.message);
    }
  } catch {
    /* fall through */
  }
  return new NeurovisionApiError(
    response.status,
    "UNKNOWN_ERROR",
    `Request failed with status ${response.status}.`,
  );
}

export function resolveAssetUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  return `${API_BASE}${path}`;
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/api/health`);
  if (!response.ok) {
    throw await parseError(response);
  }
  return (await response.json()) as HealthResponse;
}

export async function analyzeVideo(file: File): Promise<VideoAnalyzeResponse> {
  const form = new FormData();
  form.append("file", file, file.name);
  const response = await fetch(`${API_BASE}/api/videos/analyze`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    throw await parseError(response);
  }
  return (await response.json()) as VideoAnalyzeResponse;
}

// Phase 2 Job & Model API calls
export async function createJob(request: JobCreateRequest): Promise<JobResponse> {
  const response = await fetch(`${API_BASE}/api/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw await parseError(response);
  }
  return (await response.json()) as JobResponse;
}

export async function getJob(jobId: string): Promise<JobResponse> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}`);
  if (!response.ok) {
    throw await parseError(response);
  }
  return (await response.json()) as JobResponse;
}

export async function getJobResults(jobId: string): Promise<JobDetailResponse> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}/results`);
  if (!response.ok) {
    throw await parseError(response);
  }
  return (await response.json()) as JobDetailResponse;
}

export async function listJobs(): Promise<JobResponse[]> {
  const response = await fetch(`${API_BASE}/api/jobs`);
  if (!response.ok) {
    throw await parseError(response);
  }
  return (await response.json()) as JobResponse[];
}

export async function deleteJob(jobId: string): Promise<{ status: string; job_id: string }> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw await parseError(response);
  }
  return (await response.json()) as { status: string; job_id: string };
}

export async function getModelClasses(): Promise<ModelClassesResponse> {
  const response = await fetch(`${API_BASE}/api/models/classes`);
  if (!response.ok) {
    throw await parseError(response);
  }
  return (await response.json()) as ModelClassesResponse;
}

// =========================================================================
// Investigation Platform API Client
// =========================================================================

export async function getCandidates(
  videoId: string,
  frameIndex = 0,
  confidence = 0.25,
): Promise<CandidatesResponse> {
  const url = `${API_BASE}/api/videos/${videoId}/candidates?frame_index=${frameIndex}&confidence=${confidence}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw await parseError(response);
  }
  return (await response.json()) as CandidatesResponse;
}

export async function createInvestigation(
  request: InvestigationCreateRequest,
): Promise<InvestigationJobResponse> {
  const response = await fetch(`${API_BASE}/api/investigations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw await parseError(response);
  }
  return (await response.json()) as InvestigationJobResponse;
}

export async function getInvestigation(jobId: string): Promise<InvestigationJobResponse> {
  const response = await fetch(`${API_BASE}/api/investigations/${jobId}`);
  if (!response.ok) {
    throw await parseError(response);
  }
  return (await response.json()) as InvestigationJobResponse;
}

export async function getInvestigationResults(
  jobId: string,
): Promise<InvestigationJobDetailResponse> {
  const response = await fetch(`${API_BASE}/api/investigations/${jobId}/results`);
  if (!response.ok) {
    throw await parseError(response);
  }
  return (await response.json()) as InvestigationJobDetailResponse;
}

export async function updateEventReview(
  jobId: string,
  eventId: string,
  reviewStatus: ReviewStatus,
  notes?: string,
): Promise<InvestigationEvent> {
  const response = await fetch(`${API_BASE}/api/investigations/${jobId}/events/${eventId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ review_status: reviewStatus, notes }),
  });
  if (!response.ok) {
    throw await parseError(response);
  }
  return (await response.json()) as InvestigationEvent;
}

export async function deleteInvestigation(
  jobId: string,
): Promise<{ status: string; job_id: string }> {
  const response = await fetch(`${API_BASE}/api/investigations/${jobId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw await parseError(response);
  }
  return (await response.json()) as { status: string; job_id: string };
}
