import { useCallback } from 'react';
import { useDropzone, Accept } from 'react-dropzone';
import { Upload, Image, Video, X } from 'lucide-react';
import clsx from 'clsx';

interface UploadDropzoneProps {
  type: 'image' | 'video';
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
  disabled?: boolean;
}

const acceptTypes: Record<string, Accept> = {
  image: {
    'image/jpeg': ['.jpg', '.jpeg'],
    'image/png': ['.png'],
    'image/webp': ['.webp'],
  },
  video: {
    'video/mp4': ['.mp4'],
    'video/avi': ['.avi'],
    'video/quicktime': ['.mov'],
    'video/webm': ['.webm'],
  },
};

export default function UploadDropzone({
  type,
  onFileSelect,
  selectedFile,
  onClear,
  disabled = false,
}: UploadDropzoneProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onFileSelect(acceptedFiles[0]);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptTypes[type],
    multiple: false,
    disabled,
  });

  const Icon = type === 'image' ? Image : Video;

  if (selectedFile) {
    return (
      <div className="border-2 border-primary-200 bg-primary-50 rounded-xl p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-primary-100 rounded-lg">
              <Icon className="w-6 h-6 text-primary-600" />
            </div>
            <div>
              <p className="font-medium text-gray-900">{selectedFile.name}</p>
              <p className="text-sm text-gray-500">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          </div>
          <button
            onClick={onClear}
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
            disabled={disabled}
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      {...getRootProps()}
      className={clsx(
        'border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all',
        isDragActive
          ? 'border-primary-500 bg-primary-50'
          : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50',
        disabled && 'opacity-50 cursor-not-allowed'
      )}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center gap-4">
        <div
          className={clsx(
            'p-4 rounded-full',
            isDragActive ? 'bg-primary-100' : 'bg-gray-100'
          )}
        >
          <Upload
            className={clsx(
              'w-8 h-8',
              isDragActive ? 'text-primary-600' : 'text-gray-400'
            )}
          />
        </div>
        <div>
          <p className="text-lg font-medium text-gray-900">
            {isDragActive ? 'Drop the file here' : `Drop ${type} file here`}
          </p>
          <p className="text-sm text-gray-500 mt-1">
            or click to browse from your computer
          </p>
        </div>
        <p className="text-xs text-gray-400">
          {type === 'image'
            ? 'Supports: JPEG, PNG, WebP (max 100MB)'
            : 'Supports: MP4, AVI, MOV, WebM (max 100MB)'}
        </p>
      </div>
    </div>
  );
}
