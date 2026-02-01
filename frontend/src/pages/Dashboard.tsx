import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Upload, Video, Clock, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { checkHealth, HealthCheck } from '../api';

interface RecentAnalysis {
  id: string;
  type: 'image' | 'video';
  status: 'pending' | 'processing' | 'completed' | 'failed';
  verdict?: 'REAL' | 'FAKE' | 'UNCERTAIN';
  confidence?: number;
  created_at: string;
}

// Mock recent analyses for demo
const mockRecentAnalyses: RecentAnalysis[] = [
  {
    id: '1',
    type: 'image',
    status: 'completed',
    verdict: 'REAL',
    confidence: 0.94,
    created_at: new Date().toISOString(),
  },
  {
    id: '2',
    type: 'video',
    status: 'completed',
    verdict: 'FAKE',
    confidence: 0.87,
    created_at: new Date(Date.now() - 3600000).toISOString(),
  },
  {
    id: '3',
    type: 'image',
    status: 'processing',
    created_at: new Date(Date.now() - 7200000).toISOString(),
  },
];

export default function Dashboard() {
  const [health, setHealth] = useState<HealthCheck | null>(null);
  const [recentAnalyses] = useState<RecentAnalysis[]>(mockRecentAnalyses);

  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(console.error);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-success-500" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-danger-500" />;
      case 'processing':
      case 'pending':
        return <Clock className="w-5 h-5 text-primary-500 animate-pulse" />;
      default:
        return <AlertCircle className="w-5 h-5 text-gray-400" />;
    }
  };

  const getVerdictBadge = (verdict?: string) => {
    if (!verdict) return null;
    const classes = {
      REAL: 'badge-success',
      FAKE: 'badge-danger',
      UNCERTAIN: 'badge-warning',
    };
    return <span className={`badge ${classes[verdict as keyof typeof classes]}`}>{verdict}</span>;
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Welcome Section */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500 mt-1">
          Analyze media files for potential manipulation
        </p>
      </div>

      {/* Quick Actions */}
      <div className="grid md:grid-cols-2 gap-6">
        <Link to="/upload/image" className="card card-hover group">
          <div className="flex items-center gap-4">
            <div className="p-4 bg-primary-100 rounded-xl group-hover:bg-primary-200 transition-colors">
              <Upload className="w-8 h-8 text-primary-600" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Image Analysis</h2>
              <p className="text-gray-500">Upload an image for instant analysis</p>
            </div>
          </div>
        </Link>

        <Link to="/upload/video" className="card card-hover group">
          <div className="flex items-center gap-4">
            <div className="p-4 bg-purple-100 rounded-xl group-hover:bg-purple-200 transition-colors">
              <Video className="w-8 h-8 text-purple-600" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Video Analysis</h2>
              <p className="text-gray-500">Upload a video for frame-by-frame analysis</p>
            </div>
          </div>
        </Link>
      </div>

      {/* System Status */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">System Status</h2>
        <div className="flex items-center gap-3">
          <div
            className={`w-3 h-3 rounded-full ${
              health?.status === 'healthy' ? 'bg-success-500' : 'bg-gray-300'
            }`}
          />
          <span className="text-gray-600">
            {health?.status === 'healthy' ? 'All systems operational' : 'Checking status...'}
          </span>
          {health && (
            <span className="text-sm text-gray-400">v{health.version}</span>
          )}
        </div>
      </div>

      {/* Recent Analyses */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent Analyses</h2>
        
        {recentAnalyses.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <p>No recent analyses</p>
            <p className="text-sm mt-1">Upload a file to get started</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm text-gray-500 border-b border-gray-100">
                  <th className="pb-3 font-medium">Status</th>
                  <th className="pb-3 font-medium">Type</th>
                  <th className="pb-3 font-medium">Verdict</th>
                  <th className="pb-3 font-medium">Confidence</th>
                  <th className="pb-3 font-medium">Date</th>
                  <th className="pb-3 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {recentAnalyses.map((analysis) => (
                  <tr
                    key={analysis.id}
                    className="border-b border-gray-50 hover:bg-gray-50"
                  >
                    <td className="py-4">{getStatusIcon(analysis.status)}</td>
                    <td className="py-4">
                      <span className="badge badge-info capitalize">{analysis.type}</span>
                    </td>
                    <td className="py-4">{getVerdictBadge(analysis.verdict)}</td>
                    <td className="py-4 text-gray-600">
                      {analysis.confidence
                        ? `${(analysis.confidence * 100).toFixed(1)}%`
                        : '-'}
                    </td>
                    <td className="py-4 text-gray-500 text-sm">
                      {new Date(analysis.created_at).toLocaleString()}
                    </td>
                    <td className="py-4">
                      <Link
                        to={`/result/${analysis.id}`}
                        className="text-primary-600 hover:text-primary-700 text-sm font-medium"
                      >
                        View
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Load Sample Button */}
      <div className="card bg-gray-50">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-gray-900">Try with Sample Files</h3>
            <p className="text-sm text-gray-500">
              Load pre-included sample media to see how the system works
            </p>
          </div>
          <button className="btn btn-secondary">Load Sample</button>
        </div>
      </div>
    </div>
  );
}
