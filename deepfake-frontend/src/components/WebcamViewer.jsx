import React from 'react';
import { Video, Camera, X } from 'lucide-react';
import { Button } from './Button';

export const WebcamViewer = ({ 
  videoRef, 
  canvasRef, 
  isStreaming, 
  onStart, 
  onStop, 
  onCapture,
  isProcessing 
}) => {
  return (
    <div className="flex flex-col items-center">
      <div className="relative mb-4 w-full max-w-2xl">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full rounded-lg bg-black shadow-2xl"
          style={{ maxHeight: '500px', aspectRatio: '16/9' }}
        />
        <canvas ref={canvasRef} className="hidden" />
        
        {isStreaming && (
          <div className="absolute top-4 right-4 flex items-center gap-2 bg-red-600 px-3 py-1 rounded-full">
            <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
            <span className="text-white text-sm font-medium">LIVE</span>
          </div>
        )}
      </div>
      
      <div className="flex gap-4">
        {!isStreaming ? (
          <Button
            onClick={onStart}
            variant="success"
            icon={Video}
          >
            Start Camera
          </Button>
        ) : (
          <>
            <Button
              onClick={onCapture}
              disabled={isProcessing}
              variant="primary"
              icon={Camera}
            >
              Capture & Analyze
            </Button>
            <Button
              onClick={onStop}
              variant="danger"
              icon={X}
            >
              Stop Camera
            </Button>
          </>
        )}
      </div>
    </div>
  );
};
