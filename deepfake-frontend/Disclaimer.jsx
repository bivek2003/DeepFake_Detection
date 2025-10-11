import React from 'react';
import { AlertTriangle } from 'lucide-react';

export const Disclaimer = () => {
  return (
    <div className="text-center text-sm text-slate-400 mt-12 pb-8">
      <div className="flex items-center justify-center gap-2 mb-2">
        <AlertTriangle size={16} className="text-yellow-500" />
        <p className="font-semibold text-slate-300">Important Disclaimer</p>
      </div>
      <p className="mb-2">
        This tool is not 100% accurate. Always verify critical content through multiple sources.
      </p>
      <p>
        Detection technology may not catch all deepfakes, especially sophisticated ones.
      </p>
    </div>
  );
};
