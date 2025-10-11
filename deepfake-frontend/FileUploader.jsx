import React from 'react';
import { Upload } from 'lucide-react';
import { ACCEPTED_FILE_TYPES } from '../constants/api';

export const FileUploader = ({ onFileSelect, fileInputRef }) => {
  const acceptTypes = Object.values(ACCEPTED_FILE_TYPES).join(',');

  return (
    <div className="text-center">
      <input
        ref={fileInputRef}
        type="file"
        accept={acceptTypes}
        onChange={onFileSelect}
        className="hidden"
        id="file-upload"
      />
      <label
        htmlFor="file-upload"
        className="inline-flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-slate-600 rounded-lg cursor-pointer hover:border-purple-500 hover:bg-slate-700/30 transition-all"
      >
        <Upload size={48} className="mb-4 text-slate-400" />
        <p className="text-xl mb-2 text-slate-300">Click to upload</p>
        <p className="text-sm text-slate-500">
          Support for images, videos, and audio files (max 50MB)
        </p>
      </label>
    </div>
  );
};
