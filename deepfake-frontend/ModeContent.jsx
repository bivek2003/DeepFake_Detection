import React from 'react';
import { UploadMode } from './modes/UploadMode';
import { WebcamMode } from './modes/WebcamMode';
import { MODES } from '../constants/ui';

export const ModeContent = ({ 
  mode, 
  uploadProps,
  webcamProps,
  isProcessing 
}) => {
  switch (mode) {
    case MODES.UPLOAD:
      return <UploadMode {...uploadProps} />;
    case MODES.WEBCAM:
      return <WebcamMode {...webcamProps} isProcessing={isProcessing} />;
    default:
      return <UploadMode {...uploadProps} />;
  }
};
