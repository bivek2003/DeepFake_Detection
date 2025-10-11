import apiClient from './api';
import { API_ENDPOINTS } from '../constants/api';

export const predictImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await apiClient.post(API_ENDPOINTS.PREDICT_IMAGE, formData);
  return response.data;
};

export const predictVideo = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await apiClient.post(API_ENDPOINTS.PREDICT_VIDEO, formData);
  return response.data;
};

export const predictAudio = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await apiClient.post(API_ENDPOINTS.PREDICT_AUDIO, formData);
  return response.data;
};
