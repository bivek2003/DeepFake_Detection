import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { ArrowLeft, Loader2, Download } from 'lucide-react';
import VerdictCard from '../components/VerdictCard';
import HeatmapViewer from '../components/HeatmapViewer';
import TimelineChart from '../components/TimelineChart';
import FrameGallery from '../components/FrameGallery';
import { getJobResult, JobResult, getReportUrl, getErrorMessage } from '../api';

export default function ResultDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [result, setResult] = useState<JobResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;

    const fetchResult = async () => {
      try {
        const data = await getJobResult(id);
        setResult(data);
      } catch (err) {
        setError(getErrorMessage(err));
      } finally {
        setLoading(false);
      }
    };

    fetchResult();
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-8">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate(-1)}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </button>
          <h1 className="text-3xl font-bold text-gray-900">Analysis Result</h1>
        </div>

        <div className="card border-danger-200 bg-danger-50">
          <p className="text-danger-700">{error}</p>
          <Link to="/" className="btn btn-secondary mt-4">
            Return to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="space-y-8">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate(-1)}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </button>
          <h1 className="text-3xl font-bold text-gray-900">Analysis Result</h1>
        </div>

        <div className="card">
          <p className="text-gray-500">Result not found</p>
        </div>
      </div>
    );
  }

  const isVideo = result.total_frames !== null && result.total_frames > 0;

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate(-1)}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </button>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Analysis Result</h1>
            <p className="text-gray-500 mt-1">
              {isVideo ? 'Video' : 'Image'} analysis completed
            </p>
          </div>
        </div>

        {result.report_url && (
          <a
            href={getReportUrl(result.job_id)}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-primary"
          >
            <Download className="w-4 h-4" />
            Download Report
          </a>
        )}
      </div>

      {/* Verdict */}
      <VerdictCard
        verdict={result.verdict}
        confidence={result.confidence}
        status={result.status}
      />

      {/* Video-specific content */}
      {isVideo && (
        <>
          {result.chart_data && (
            <TimelineChart data={result.chart_data.timeline} />
          )}
          <FrameGallery frames={result.suspicious_frames} />
        </>
      )}

      {/* Heatmap: image uses heatmap_url; video uses first suspicious frame overlay */}
      <HeatmapViewer
        heatmapUrl={
          result.heatmap_url ||
          result.suspicious_frames[0]?.overlay_url ||
          null
        }
        title={isVideo ? 'Suspicious frame heatmap' : 'Heatmap Analysis'}
      />

      {/* Metadata */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis Details</h3>
        <dl className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <dt className="text-sm text-gray-500">Job ID</dt>
            <dd className="font-mono text-sm text-gray-900 truncate">{result.job_id}</dd>
          </div>
          <div>
            <dt className="text-sm text-gray-500">Model Version</dt>
            <dd className="text-gray-900">{result.model_version}</dd>
          </div>
          <div>
            <dt className="text-sm text-gray-500">Processing Time</dt>
            <dd className="text-gray-900">
              {result.runtime_ms ? `${result.runtime_ms}ms` : '-'}
            </dd>
          </div>
          <div>
            <dt className="text-sm text-gray-500">Device</dt>
            <dd className="text-gray-900">{result.device}</dd>
          </div>
          {isVideo && (
            <>
              <div>
                <dt className="text-sm text-gray-500">Total Frames</dt>
                <dd className="text-gray-900">{result.total_frames}</dd>
              </div>
              <div>
                <dt className="text-sm text-gray-500">Analyzed Frames</dt>
                <dd className="text-gray-900">{result.analyzed_frames}</dd>
              </div>
            </>
          )}
          <div>
            <dt className="text-sm text-gray-500">Created</dt>
            <dd className="text-gray-900">
              {new Date(result.created_at).toLocaleString()}
            </dd>
          </div>
          {result.completed_at && (
            <div>
              <dt className="text-sm text-gray-500">Completed</dt>
              <dd className="text-gray-900">
                {new Date(result.completed_at).toLocaleString()}
              </dd>
            </div>
          )}
          <div className="col-span-2 md:col-span-4">
            <dt className="text-sm text-gray-500">File Hash (SHA256)</dt>
            <dd className="font-mono text-xs text-gray-900 break-all">
              {result.sha256}
            </dd>
          </div>
        </dl>
      </div>

      {/* Navigation */}
      <div className="flex gap-4">
        <Link to="/" className="btn btn-secondary">
          Back to Dashboard
        </Link>
        <Link
          to={isVideo ? '/upload/video' : '/upload/image'}
          className="btn btn-primary"
        >
          New Analysis
        </Link>
      </div>
    </div>
  );
}
