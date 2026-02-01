import React from 'react';
import { AlertTriangle } from 'lucide-react';
import { FrameScore, getAssetUrl } from '../api';

interface FrameGalleryProps {
  frames: FrameScore[];
  title?: string;
}

export default function FrameGallery({
  frames,
  title = 'Suspicious Frames',
}: FrameGalleryProps) {
  if (!frames || frames.length === 0) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
        <div className="bg-gray-100 rounded-lg p-8 text-center">
          <p className="text-gray-500">No suspicious frames detected</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-4">
        <AlertTriangle className="w-5 h-5 text-amber-500" />
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {frames.map((frame) => (
          <div
            key={frame.frame_index}
            className="relative rounded-lg overflow-hidden border-2 border-gray-200 hover:border-primary-400 transition-colors"
          >
            {frame.overlay_url ? (
              <img
                src={getAssetUrl(frame.overlay_url)}
                alt={`Frame ${frame.frame_index}`}
                className="w-full h-auto aspect-video object-cover"
              />
            ) : (
              <div className="w-full aspect-video bg-gray-100 flex items-center justify-center">
                <span className="text-gray-400 text-sm">No preview</span>
              </div>
            )}
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-2">
              <div className="flex items-center justify-between text-white text-xs">
                <span>t={frame.timestamp.toFixed(1)}s</span>
                <span
                  className={`px-1.5 py-0.5 rounded ${
                    frame.score > 0.7
                      ? 'bg-red-500'
                      : frame.score > 0.5
                      ? 'bg-amber-500'
                      : 'bg-green-500'
                  }`}
                >
                  {(frame.score * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <p className="mt-4 text-sm text-gray-500">
        These frames showed the highest fake probability scores during analysis. Red
        indicates high confidence of manipulation.
      </p>
    </div>
  );
}
