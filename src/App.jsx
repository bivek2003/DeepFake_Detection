import React from 'react';
import {
  Header,
  ModeSelector,
  ContentContainer,
  ModeContent,
  LoadingSpinner,
  ErrorAlert,
  ResultCard,
  Disclaimer,
} from './components';
import { useMode, useWebcam, usePrediction, useFileUpload, useKeyboardShortcuts } from './hooks';
import { MODES, MESSAGES } from './constants/ui';

function App() {
  const { mode, switchMode } = useMode();
  const webcam = useWebcam();
  const prediction = usePrediction();
  const upload = useFileUpload();

  const handleModeChange = (newMode) => {
    switchMode(newMode);
    if (newMode !== MODES.WEBCAM) {
      webcam.stop();
    }
    prediction.reset();
    upload.reset();
  };

  const handleFileUpload = async (event) => {
    const fileData = upload.handleFileSelect(event);
    if (fileData) {
      await prediction.predict(fileData.file, fileData.type);
    }
  };

  const handleWebcamCapture = async () => {
    const frame = await webcam.capture();
    if (frame) {
      await prediction.predict(frame, 'image');
    }
  };

  const handleReset = () => {
    prediction.reset();
    upload.reset();
  };

  // Keyboard shortcuts
  useKeyboardShortcuts([
    { key: 'u', callback: () => handleModeChange(MODES.UPLOAD) },
    { key: 'w', callback: () => handleModeChange(MODES.WEBCAM) },
    { key: 'r', callback: handleReset },
    { key: 'Escape', callback: () => webcam.isStreaming && webcam.stop() },
  ]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        <Header />
        
        <ModeSelector 
          currentMode={mode} 
          onModeChange={handleModeChange} 
        />

        <ContentContainer>
          <ModeContent
            mode={mode}
            uploadProps={{
              onFileSelect: handleFileUpload,
              fileInputRef: upload.fileInputRef,
            }}
            webcamProps={{
              webcam,
              onCapture: handleWebcamCapture,
            }}
            isProcessing={prediction.isProcessing}
          />
        </ContentContainer>

        {prediction.isProcessing && (
          <LoadingSpinner message={MESSAGES.PROCESSING} />
        )}

        <ErrorAlert 
          message={prediction.error || upload.error || webcam.error}
          onDismiss={handleReset}
        />

        <ResultCard 
          result={prediction.result} 
          onReset={handleReset} 
        />

        <Disclaimer />
      </div>
    </div>
  );
}

export default App;
