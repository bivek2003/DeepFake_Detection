export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  PREDICT_IMAGE: '/predict/image',
  PREDICT_VIDEO: '/predict/video',
  PREDICT_AUDIO: '/predict/audio',
  PREDICT_STREAM: '/predict/stream',
};

export const FILE_TYPES = {
  IMAGE: 'image',
  VIDEO: 'video',
  AUDIO: 'audio',
};

export const ACCEPTED_FILE_TYPES = {
  IMAGE: 'image/*',
  VIDEO: 'video/*',
  AUDIO: 'audio/*',
};
