import axios, { AxiosError } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

// Types
export interface HealthCheck {
  status: string;
  version: string;
  timestamp: string;
}

export interface ImageAnalysisResult {
  id: string;
  verdict: 'REAL' | 'FAKE' | 'UNCERTAIN';
  confidence: number;
  heatmap_url: string | null;
  sha256: string;
  model_version: string;
  runtime_ms: number;
  device: string;
  created_at: string;
  disclaimer: string;
}

export interface VideoAnalysisResponse {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  message: string;
}

export interface JobStatus {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string | null;
  created_at: string;
  updated_at: string | null;
  error: string | null;
}

export interface FrameScore {
  frame_index: number;
  timestamp: number;
  score: number;
  overlay_url: string | null;
}

export interface ChartData {
  timeline: Array<{ timestamp: number; score: number; frame: number }>;
  distribution: {
    buckets: Array<{ range: string; count: number }>;
    mean: number;
    max: number;
    min: number;
  };
}

export interface JobResult {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  verdict: 'REAL' | 'FAKE' | 'UNCERTAIN' | null;
  confidence: number | null;
  sha256: string;
  model_version: string;
  runtime_ms: number | null;
  device: string;
  created_at: string;
  completed_at: string | null;
  total_frames: number | null;
  analyzed_frames: number | null;
  frame_scores: FrameScore[];
  suspicious_frames: FrameScore[];
  chart_data: ChartData | null;
  report_url: string | null;
  timeline_chart_url: string | null;
  disclaimer: string;
}

export interface ModelInfo {
  model_name: string;
  model_version: string;
  commit_hash: string | null;
  calibration_method: string | null;
  demo_mode: boolean;
  device: string;
  metrics: Record<string, unknown> | null;
}

// API Functions
export async function checkHealth(): Promise<HealthCheck> {
  const response = await api.get<HealthCheck>('/api/v1/healthz');
  return response.data;
}

export async function analyzeImage(file: File): Promise<ImageAnalysisResult> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<ImageAnalysisResult>(
    '/api/v1/analyze/image',
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
    }
  );
  return response.data;
}

export async function analyzeVideo(file: File): Promise<VideoAnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<VideoAnalysisResponse>(
    '/api/v1/analyze/video',
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
    }
  );
  return response.data;
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const response = await api.get<JobStatus>(`/api/v1/jobs/${jobId}`);
  return response.data;
}

export async function getJobResult(jobId: string): Promise<JobResult> {
  const response = await api.get<JobResult>(`/api/v1/jobs/${jobId}/result`);
  return response.data;
}

export async function getModelInfo(): Promise<ModelInfo> {
  const response = await api.get<ModelInfo>('/api/v1/model/info');
  return response.data;
}

export function getReportUrl(jobId: string): string {
  return `${API_BASE_URL}/api/v1/reports/${jobId}.pdf`;
}

export function getAssetUrl(path: string): string {
  if (path.startsWith('/')) {
    return `${API_BASE_URL}${path}`;
  }
  return `${API_BASE_URL}/api/v1/assets/${path}`;
}

// Error handling
export function getErrorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    if (error.response?.data?.detail) {
      return error.response.data.detail;
    }
    if (error.message) {
      return error.message;
    }
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred';
}
