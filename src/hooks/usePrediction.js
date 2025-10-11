import { useState } from 'react';
import { predictImage, predictVideo, predictAudio } from '../services/predictionService';
import { logPrediction, logError } from '../utils/analytics';

export const usePrediction = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const predict = async (file, type) => {
    setIsProcessing(true);
    setError(null);
    setResult(null);

    const startTime = performance.now();

    try {
      let response;
      
      switch (type) {
        case 'image':
          response = await predictImage(file);
          break;
        case 'video':
          response = await predictVideo(file);
          break;
        case 'audio':
          response = await predictAudio(file);
          break;
        default:
          throw new Error('Invalid file type');
      }
      
      const duration = performance.now() - startTime;
      logPrediction(type, response, duration);
      
      setResult(response);
      return response;
    } catch (err) {
      const errorMessage = err.response?.data?.message || err.message || 'Analysis failed';
      setError(errorMessage);
      logError(err, { type, context: 'prediction' });
      return null;
    } finally {
      setIsProcessing(false);
    }
  };

  const reset = () => {
    setResult(null);
    setError(null);
  };

  return {
    isProcessing,
    result,
    error,
    predict,
    reset,
  };
};
