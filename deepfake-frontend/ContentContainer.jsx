import React from 'react';

export const ContentContainer = ({ children, className = '' }) => {
  return (
    <div className={`bg-slate-800/80 backdrop-blur-sm rounded-xl shadow-2xl p-8 mb-8 border border-slate-700 ${className}`}>
      {children}
    </div>
  );
};
