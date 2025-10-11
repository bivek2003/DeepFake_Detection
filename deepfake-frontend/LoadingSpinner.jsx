import React from 'react';

export const LoadingSpinner = ({ message = 'Processing...' }) => {
  return (
    <div className="text-center py-8">
      <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mb-4"></div>
      <p className="text-slate-300">{message}</p>
    </div>
  );
};
