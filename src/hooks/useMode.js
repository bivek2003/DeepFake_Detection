import { useState } from 'react';
import { MODES } from '../constants/ui';

export const useMode = (initialMode = MODES.UPLOAD) => {
  const [mode, setMode] = useState(initialMode);

  const switchMode = (newMode) => {
    setMode(newMode);
  };

  const isUploadMode = mode === MODES.UPLOAD;
  const isWebcamMode = mode === MODES.WEBCAM;
  const isAudioMode = mode === MODES.AUDIO;

  return {
    mode,
    switchMode,
    isUploadMode,
    isWebcamMode,
    isAudioMode,
  };
};
