import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Upload, Video, Clock, CheckCircle, XCircle, AlertCircle, Loader2 } from 'lucide-react';
import { checkHealth, getRecentAnalyses, HealthCheck, AnalysisListItem, getModels, switchModel, ModelItem } from '../api';

export default function Dashboard() {
  const [health, setHealth] = useState<HealthCheck | null>(null);
  const [recentAnalyses, setRecentAnalyses] = useState<AnalysisListItem[]>([]);
  const [loadingAnalyses, setLoadingAnalyses] = useState(true);
  const [models, setModels] = useState<ModelItem[]>([]);
  const [switchingModel, setSwitchingModel] = useState<string | null>(null);

  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(console.error);

    loadModels();
  }, []);

  useEffect(() => {
    getRecentAnalyses(20)
      .then(setRecentAnalyses)
      .catch(() => setRecentAnalyses([]))
      .finally(() => setLoadingAnalyses(false));
  }, []);

  const loadModels = () => {
    getModels()
      .then(setModels)
      .catch(console.error);
  };

  const handleSwitchModel = async (filename: string) => {
    try {
      setSwitchingModel(filename);
      await switchModel(filename);
      // Wait a moment for reload
      await new Promise(resolve => setTimeout(resolve, 2000));
      await loadModels(); // Refresh status
      await checkHealth(); // Refresh health/version info
    } catch (error) {
      console.error('Failed to switch model:', error);
      alert('Failed to switch model. Check console for details.');
    } finally {
      setSwitchingModel(null);
    }
  };

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
        <Link to="/upload/image" className="card card-hover group relative overflow-hidden border-0 shadow-lg">
          <div className="absolute inset-0 bg-gradient-to-br from-primary-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative flex items-center gap-4 z-10">
            <div className="p-4 bg-primary-100 rounded-xl group-hover:bg-primary-600 group-hover:text-white transition-all duration-300">
              <Upload className="w-8 h-8 text-primary-600 group-hover:text-white" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Image Analysis</h2>
              <p className="text-gray-500">Upload an image for instant analysis</p>
            </div>
          </div>
        </Link>

        <Link to="/upload/video" className="card card-hover group relative overflow-hidden border-0 shadow-lg">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative flex items-center gap-4 z-10">
            <div className="p-4 bg-purple-100 rounded-xl group-hover:bg-purple-600 group-hover:text-white transition-all duration-300">
              <Video className="w-8 h-8 text-purple-600 group-hover:text-white" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Video Analysis</h2>
              <p className="text-gray-500">Upload a video for frame-by-frame analysis</p>
            </div>
          </div>
        </Link>
      </div>

      {/* System Status and Model Config */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="card glass relative overflow-hidden">
          <div className="absolute top-0 right-0 p-6 opacity-10">
            <CheckCircle className="w-24 h-24" />
          </div>
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            System Status
          </h2>
          <div className="flex items-center gap-3">
            <div className="relative">
              <div
                className={`w-3 h-3 rounded-full ${health?.status === 'healthy' ? 'bg-success-500' : 'bg-gray-300'
                  }`}
              />
              {health?.status === 'healthy' && (
                <div className="absolute insert-0 w-3 h-3 bg-success-500 rounded-full animate-ping opacity-75" />
              )}
            </div>

            <span className="text-gray-700 font-medium">
              {health?.status === 'healthy' ? 'All systems operational' : 'Checking status...'}
            </span>
            {health && (
              <span className="badge badge-info ml-auto">v{health.version}</span>
            )}
          </div>
        </div>

        {/* Model Configuration */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            Model Configuration
          </h2>
          <div className="space-y-3">
            {models.length === 0 ? (
              <div className="text-sm text-gray-500">Loading models...</div>
            ) : (
              models.map(model => (
                <div
                  key={model.id}
                  className={`p-3 rounded-lg border ${model.active ? 'border-primary-500 bg-primary-50' : 'border-gray-200 hover:border-gray-300'
                    } transition-colors flex items-center justify-between`}
                >
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-medium text-gray-900">{model.name}</h3>
                      {model.active && (
                        <span className="badge badge-primary text-xs">Active</span>
                      )}
                      <span className="text-xs text-gray-500">({model.size_mb} MB)</span>
                    </div>
                    <p className="text-sm text-gray-500 mt-1">{model.description}</p>
                  </div>

                  {!model.active && (
                    <button
                      onClick={() => handleSwitchModel(model.id)}
                      disabled={!!switchingModel}
                      className="btn btn-secondary py-1 px-3 text-xs whitespace-nowrap"
                    >
                      {switchingModel === model.id ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        'Activate'
                      )}
                    </button>
                  )}
                </div>
              ))
            )}

            <div className="text-xs text-gray-400 mt-2">
              <p>M12: Recommended for NVIDIA GPUs (RTX 3060+)</p>
              <p>M8: Recommended for CPU or older GPUs</p>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Analyses */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent Analyses</h2>

        {loadingAnalyses ? (
          <div className="text-center py-8 text-gray-500">
            <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2 text-primary-500" />
            <p>Loading history...</p>
          </div>
        ) : recentAnalyses.length === 0 ? (
          <div className="text-center py-12 bg-gray-50 rounded-lg border border-dashed border-gray-200">
            <div className="mx-auto w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mb-3">
              <Upload className="w-6 h-6 text-gray-400" />
            </div>
            <p className="text-gray-900 font-medium">No recent analyses</p>
            <p className="text-sm text-gray-500 mt-1">Upload a file to get started</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm text-gray-500 border-b border-gray-100">
                  <th className="pb-3 font-medium pl-2">Status</th>
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
                    className="border-b border-gray-50 hover:bg-gray-50 transition-colors"
                  >
                    <td className="py-4 pl-2">{getStatusIcon(analysis.status)}</td>
                    <td className="py-4">
                      <span className={`badge ${analysis.type === 'image' ? 'badge-info' : 'bg-purple-100 text-purple-800 border-purple-200'} capitalize`}>
                        {analysis.type}
                      </span>
                    </td>
                    <td className="py-4">{getVerdictBadge(analysis.verdict)}</td>
                    <td className="py-4 text-gray-700 font-medium">
                      {analysis.confidence
                        ? `${(analysis.confidence * 100).toFixed(1)}%`
                        : '-'}
                    </td>
                    <td className="py-4 text-gray-500 text-sm">
                      {new Date(analysis.created_at).toLocaleDateString()}
                    </td>
                    <td className="py-4 text-right pr-2">
                      <Link
                        to={`/result/${analysis.id}`}
                        className="btn btn-secondary py-1 px-3 text-xs"
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
    </div>
  );
}
