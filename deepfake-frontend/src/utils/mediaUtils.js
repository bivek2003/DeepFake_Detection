export const getFileType = (file) => {
  if (file.type.startsWith('image')) return 'image';
  if (file.type.startsWith('video')) return 'video';
  if (file.type.startsWith('audio')) return 'audio';
  return null;
};

export const validateFile = (file, maxSize = 50 * 1024 * 1024) => {
  if (!file) return { valid: false, error: 'No file selected' };
  
  const type = getFileType(file);
  if (!type) return { valid: false, error: 'Unsupported file type' };
  
  if (file.size > maxSize) {
    return { valid: false, error: 'File size exceeds 50MB limit' };
  }
  
  return { valid: true, type };
};

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
};
