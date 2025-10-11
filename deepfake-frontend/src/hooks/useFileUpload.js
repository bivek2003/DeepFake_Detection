import { useState, useRef } from 'react';
import { validateFile } from '../utils/mediaUtils';

export const useFileUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    
    if (!file) {
      setSelectedFile(null);
      return null;
    }

    const validation = validateFile(file);
    
    if (!validation.valid) {
      setError(validation.error);
      setSelectedFile(null);
      return null;
    }

    setError(null);
    setSelectedFile(file);
    return { file, type: validation.type };
  };

  const reset = () => {
    setSelectedFile(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return {
    selectedFile,
    error,
    fileInputRef,
    handleFileSelect,
    reset,
  };
};
