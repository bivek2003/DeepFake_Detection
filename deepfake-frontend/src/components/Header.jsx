import React from 'react';

export const Header = () => {
  return (
    <div className="text-center mb-12">
      <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
        Deepfake Detector
      </h1>
      <p className="text-slate-300 text-lg">
        Upload media or use your webcam to detect AI-generated content
      </p>
    </div>
  );
};
