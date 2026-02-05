import { useState } from 'react';
import { Eye, EyeOff } from 'lucide-react';
import { getAssetUrl } from '../api';

interface HeatmapViewerProps {
  originalUrl?: string;
  heatmapUrl: string | null;
  title?: string;
}

export default function HeatmapViewer({
  originalUrl,
  heatmapUrl,
  title = 'Heatmap Analysis',
}: HeatmapViewerProps) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [imgError, setImgError] = useState(false);

  if (!heatmapUrl) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
        <div className="bg-gray-100 rounded-lg p-8 text-center">
          <p className="text-gray-500">No heatmap available</p>
        </div>
      </div>
    );
  }

  const displayUrl = getAssetUrl(heatmapUrl);

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        <button
          onClick={() => setShowHeatmap(!showHeatmap)}
          className="btn btn-secondary text-sm"
          type="button"
        >
          {showHeatmap ? (
            <>
              <EyeOff className="w-4 h-4" />
              Hide Overlay
            </>
          ) : (
            <>
              <Eye className="w-4 h-4" />
              Show Overlay
            </>
          )}
        </button>
      </div>

      <div className="relative rounded-lg overflow-hidden bg-gray-100 min-h-[200px]">
        {imgError ? (
          <div className="flex items-center justify-center p-8 text-gray-500">
            <p>Could not load heatmap image</p>
          </div>
        ) : (
          <img
            src={showHeatmap ? displayUrl : originalUrl || displayUrl}
            alt="Analysis visualization"
            className="w-full h-auto max-h-[500px] object-contain"
            onError={() => setImgError(true)}
            crossOrigin="anonymous"
          />
        )}
      </div>

      <div className="mt-4 text-sm text-gray-500">
        <p>
          <strong>Legend:</strong> Red/Yellow areas indicate regions that contributed most
          to the detection result. Blue/Green areas indicate lower attention.
        </p>
      </div>
    </div>
  );
}
