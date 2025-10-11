import React from 'react';
import { WebcamViewer } from '../WebcamViewer';

export const WebcamMode = ({ 
  webcam,
  isProcessing,
  onCapture 
}) => {
  return (
    <WebcamViewer
      videoRef={webcam.videoRef}
      canvasRef={webcam.canvasRef}
      isStreaming={webcam.isStreaming}
      onStart={webcam.start}
      onStop={webcam.stop}
      onCapture={onCapture}
      isProcessing={isProcessing}
    />
  );
};
