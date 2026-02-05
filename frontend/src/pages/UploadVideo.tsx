import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2, ArrowLeft, RefreshCw } from 'lucide-react';
import UploadDropzone from '../components/UploadDropzone';
import VerdictCard from '../components/VerdictCard';
import TimelineChart from '../components/TimelineChart';
import FrameGallery from '../components/FrameGallery';
import {
  analyzeVideo,
  getJobStatus,
  getJobResult,
  JobStatus,
  JobResult,
  getReportUrl,
  getErrorMessage,
} from '../api';

export default function UploadVideo() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [result, setResult] = useState<JobResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollIntervalRef = useRef<number | null>(null);

  // Poll for job status
  useEffect(() => {
    if (!jobId || result) return;

    const pollStatus = async () => {
      try {
        const status = await getJobStatus(jobId);
        setJobStatus(status);

        if (status.status === 'completed') {
          // Fetch full results
          const fullResult = await getJobResult(jobId);
          setResult(fullResult);
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
          }
        } else if (status.status === 'failed') {
          setError(status.error || 'Analysis failed');
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
          }
        }
      } catch (err) {
        console.error('Failed to poll status:', err);
      }
    };

    // Initial poll
    pollStatus();

    // Set up interval
    pollIntervalRef.current = window.setInterval(pollStatus, 2000);

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [jobId, result]);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setJobId(null);
    setJobStatus(null);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setJobId(null);
    setJobStatus(null);
    setResult(null);
    setError(null);
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;

    setIsSubmitting(true);
    setError(null);

    try {
      const response = await analyzeVideo(selectedFile);
      setJobId(response.job_id);
      setJobStatus({
        job_id: response.job_id,
        status: response.status,
        progress: 0,
        message: response.message,
        created_at: new Date().toISOString(),
        updated_at: null,
        error: null,
      });
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  const getProgressPercent = () => {
    if (!jobStatus) return 0;
    return Math.round(jobStatus.progress * 100);
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate('/')}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <ArrowLeft className="w-5 h-5 text-gray-600" />
        </button>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Video Analysis</h1>
          <p className="text-gray-500 mt-1">
            Upload a video for frame-by-frame deepfake analysis
          </p>
        </div>
      </div>

      {/* Upload Section */}
      {!jobId && (
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload Video</h2>
          <UploadDropzone
            type="video"
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
            onClear={handleClear}
            disabled={isSubmitting}
          />

          {selectedFile && (
            <div className="mt-6 flex justify-end">
              <button
                onClick={handleSubmit}
                disabled={isSubmitting}
                className="btn btn-primary"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Submitting...
                  </>
                ) : (
                  'Start Analysis'
                )}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Processing Status */}
      {jobId && !result && jobStatus?.status !== 'failed' && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">Processing Video</h2>
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <RefreshCw className="w-4 h-4 animate-spin" />
              Updating...
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-600 capitalize">
                {jobStatus?.status === 'pending' ? 'Queued' : 'Processing'}
              </span>
              <span className="font-medium">{getProgressPercent()}%</span>
            </div>

            <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-500 transition-all duration-300"
                style={{ width: `${getProgressPercent()}%` }}
              />
            </div>

            {jobStatus?.message && (
              <p className="text-sm text-gray-500">{jobStatus.message}</p>
            )}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="card border-danger-200 bg-danger-50">
          <p className="text-danger-700">{error}</p>
          <button onClick={handleClear} className="btn btn-secondary mt-4">
            Try Again
          </button>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          <VerdictCard
            verdict={result.verdict}
            confidence={result.confidence}
            status={result.status}
          />

          {/* Timeline Chart */}
          {result.chart_data && (
            <TimelineChart data={result.chart_data.timeline} />
          )}

          {/* Suspicious Frames */}
          <FrameGallery frames={result.suspicious_frames} />

          {/* Metadata */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Analysis Details
            </h3>
            <dl className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <dt className="text-sm text-gray-500">Job ID</dt>
                <dd className="font-mono text-sm text-gray-900 truncate">
                  {result.job_id}
                </dd>
              </div>
              <div>
                <dt className="text-sm text-gray-500">Total Frames</dt>
                <dd className="text-gray-900">{result.total_frames || '-'}</dd>
              </div>
              <div>
                <dt className="text-sm text-gray-500">Analyzed Frames</dt>
                <dd className="text-gray-900">{result.analyzed_frames || '-'}</dd>
              </div>
              <div>
                <dt className="text-sm text-gray-500">Processing Time</dt>
                <dd className="text-gray-900">
                  {result.runtime_ms ? `${result.runtime_ms}ms` : '-'}
                </dd>
              </div>
              <div>
                <dt className="text-sm text-gray-500">Model Version</dt>
                <dd className="text-gray-900">{result.model_version}</dd>
              </div>
              <div>
                <dt className="text-sm text-gray-500">Device</dt>
                <dd className="text-gray-900">{result.device}</dd>
              </div>
              <div className="col-span-2">
                <dt className="text-sm text-gray-500">File Hash (SHA256)</dt>
                <dd className="font-mono text-xs text-gray-900 break-all">
                  {result.sha256}
                </dd>
              </div>
            </dl>
          </div>

          {/* Actions */}
          <div className="flex gap-4">
            {result.report_url && (
              <a
                href={getReportUrl(result.job_id)}
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-primary"
              >
                Download PDF Report
              </a>
            )}
            <button onClick={handleClear} className="btn btn-secondary">
              Analyze Another Video
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
