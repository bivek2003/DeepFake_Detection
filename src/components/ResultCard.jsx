import React from 'react';
import { CheckCircle, AlertCircle } from 'lucide-react';
import { Button } from './Button';
import { ConfidenceBar } from './ConfidenceBar';
import { PREDICTION_TYPES } from '../constants/ui';

export const ResultCard = ({ result, onReset }) => {
  if (!result) return null;

  const isReal = result.prediction === PREDICTION_TYPES.REAL;

  return (
    <div className={`rounded-xl p-6 mb-8 animate-fade-in ${
      isReal 
        ? 'bg-green-900/50 border-2 border-green-500' 
        : 'bg-red-900/50 border-2 border-red-500'
    }`}>
      <div className="flex items-center gap-4 mb-4">
        {isReal ? (
          <CheckCircle size={48} className="text-green-400 flex-shrink-0" />
        ) : (
          <AlertCircle size={48} className="text-red-400 flex-shrink-0" />
        )}
        <div className="flex-1">
          <h2 className="text-3xl font-bold">
            {isReal ? 'Authentic Content' : 'Deepfake Detected'}
          </h2>
          <ConfidenceBar confidence={result.confidence} isReal={isReal} />
        </div>
      </div>
      
      {result.explanation && (
        <div className="mt-4 p-4 bg-slate-800/50 rounded-lg border border-slate-700">
          <h3 className="text-sm font-semibold text-slate-400 mb-2">Analysis Details</h3>
          <p className="text-sm text-slate-300">{result.explanation}</p>
        </div>
      )}

      <Button
        onClick={onReset}
        variant="neutral"
        className="mt-4"
      >
        Analyze Another
      </Button>
    </div>
  );
};
