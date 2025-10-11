export const startCamera = async (constraints = {}) => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, ...constraints.video },
      audio: false,
    });
    return { stream, error: null };
  } catch (error) {
    let errorMessage = 'Unable to access camera';
    
    if (error.name === 'NotAllowedError') {
      errorMessage = 'Camera access denied. Please grant permission.';
    } else if (error.name === 'NotFoundError') {
      errorMessage = 'No camera found on this device.';
    }
    
    return { stream: null, error: errorMessage };
  }
};

export const stopCamera = (stream) => {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
};

export const captureFrame = (videoElement, canvasElement) => {
  if (!videoElement || !canvasElement) return null;
  
  const canvas = canvasElement;
  const video = videoElement;
  
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  
  return new Promise((resolve) => {
    canvas.toBlob(resolve, 'image/jpeg', 0.95);
  });
};
