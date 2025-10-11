import { useState, useEffect, useRef } from 'react';
import { startCamera, stopCamera, captureFrame } from '../utils/webcamUtils';

export const useWebcam = () => {
  const [stream, setStream] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    return () => {
      if (stream) {
        stopCamera(stream);
      }
    };
  }, [stream]);

  const start = async () => {
    const { stream: mediaStream, error: err } = await startCamera();
    
    if (err) {
      setError(err);
      return false;
    }
    
    setStream(mediaStream);
    setIsStreaming(true);
    setError(null);
    
    if (videoRef.current) {
      videoRef.current.srcObject = mediaStream;
    }
    
    return true;
  };

  const stop = () => {
    stopCamera(stream);
    setStream(null);
    setIsStreaming(false);
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const capture = async () => {
    return await captureFrame(videoRef.current, canvasRef.current);
  };

  return {
    videoRef,
    canvasRef,
    isStreaming,
    error,
    start,
    stop,
    capture,
  };
};
