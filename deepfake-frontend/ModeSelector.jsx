import React from 'react';
import { Camera, Upload } from 'lucide-react';
import { Button } from './Button';
import { MODES } from '../constants/ui';

export const ModeSelector = ({ currentMode, onModeChange }) => {
  return (
    <div className="flex justify-center gap-4 mb-8">
      <Button
        onClick={() => onModeChange(MODES.UPLOAD)}
        variant={currentMode === MODES.UPLOAD ? 'primary' : 'neutral'}
        icon={Upload}
      >
        Upload File
      </Button>
      <Button
        onClick={() => onModeChange(MODES.WEBCAM)}
        variant={currentMode === MODES.WEBCAM ? 'primary' : 'neutral'}
        icon={Camera}
      >
        Webcam
      </Button>
    </div>
  );
};
