import React from 'react';
import { AlertCircle, X } from 'lucide-react';

export const ErrorAlert = ({ message, onDismiss }) => {
  if (!message) return null;

  return (
    <div className="bg-red-900/50 border-2 border-red-500 rounded-xl p-6 mb-8 animate-fade-in">
      <div className="flex items-center gap-4">
        <AlertCircle size={32} className="text-red-400 flex-shrink-0" />
        <div className="flex-1">
          <h3 className="text-xl font-bold mb-1">Error</h3>
          <p className="text-slate-300">{message}</p>
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="text-slate-400 hover:text-white transition flex-shrink-0"
            aria-label="Dismiss error"
          >
            <X size={24} />
          </button>
        )}
      </div>
    </div>
  );
};
