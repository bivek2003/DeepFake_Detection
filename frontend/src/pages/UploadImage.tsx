import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2, ArrowLeft } from 'lucide-react';
import UploadDropzone from '../components/UploadDropzone';
import VerdictCard from '../components/VerdictCard';
import HeatmapViewer from '../components/HeatmapViewer';
import { analyzeImage, ImageAnalysisResult, getErrorMessage } from '../api';

export default function UploadImage() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<ImageAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const analysisResult = await analyzeImage(selectedFile);
      setResult(analysisResult);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setIsAnalyzing(false);
    }
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
          <h1 className="text-3xl font-bold text-gray-900">Image Analysis</h1>
          <p className="text-gray-500 mt-1">
            Upload an image to analyze for potential manipulation
          </p>
        </div>
      </div>

      {/* Upload Section */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload Image</h2>
        <UploadDropzone
          type="image"
          onFileSelect={handleFileSelect}
          selectedFile={selectedFile}
          onClear={handleClear}
          disabled={isAnalyzing}
        />

        {selectedFile && !result && (
          <div className="mt-6 flex justify-end">
            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className="btn btn-primary"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                'Analyze Image'
              )}
            </button>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="card border-danger-200 bg-danger-50">
          <p className="text-danger-700">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          <VerdictCard
            verdict={result.verdict}
            confidence={result.confidence}
            status="completed"
          />

          <HeatmapViewer heatmapUrl={result.heatmap_url} />

          {/* Metadata */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Analysis Details
            </h3>
            <dl className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <dt className="text-sm text-gray-500">Analysis ID</dt>
                <dd className="font-mono text-sm text-gray-900 truncate">
                  {result.id}
                </dd>
              </div>
              <div>
                <dt className="text-sm text-gray-500">Model Version</dt>
                <dd className="text-gray-900">{result.model_version}</dd>
              </div>
              <div>
                <dt className="text-sm text-gray-500">Processing Time</dt>
                <dd className="text-gray-900">{result.runtime_ms}ms</dd>
              </div>
              <div>
                <dt className="text-sm text-gray-500">Device</dt>
                <dd className="text-gray-900">{result.device}</dd>
              </div>
              <div className="col-span-2 md:col-span-4">
                <dt className="text-sm text-gray-500">File Hash (SHA256)</dt>
                <dd className="font-mono text-xs text-gray-900 break-all">
                  {result.sha256}
                </dd>
              </div>
            </dl>
          </div>

          {/* Actions */}
          <div className="flex gap-4">
            <button onClick={handleClear} className="btn btn-secondary">
              Analyze Another Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
