import React from 'react';
import { FileUploader } from '../FileUploader';

export const UploadMode = ({ onFileSelect, fileInputRef }) => {
  return (
    <FileUploader 
      onFileSelect={onFileSelect}
      fileInputRef={fileInputRef}
    />
  );
};
