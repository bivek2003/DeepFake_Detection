import React from 'react';

export const ConfidenceBar = ({ confidence, isReal }) => {
  const percentage = (confidence * 100).toFixed(1);
  
  return (
    <div className="mt-4">
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm text-slate-300">Confidence Level</span>
        <span className="text-sm font-bold">{percentage}%</span>
      </div>
      <div className="w-full bg-slate-700 rounded-full h-3 overflow-hidden">
        <div
          className={`h-full transition-all duration-700 ease-out ${
            isReal ? 'bg-gradient-to-r from-green-500 to-green-400' : 'bg-gradient-to-r from-red-500 to-red-400'
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};
