import React from 'react';
import { AlertTriangle } from 'lucide-react';

export default function DemoBanner() {
  return (
    <div className="bg-amber-50 border-b border-amber-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2">
        <div className="flex items-center justify-center gap-2 text-sm text-amber-800">
          <AlertTriangle className="w-4 h-4" />
          <span>
            <strong>Demo Mode:</strong> Running without real model weights. Outputs are
            deterministic but not real predictions.
          </span>
        </div>
      </div>
    </div>
  );
}
