import React from 'react';
import { CheckCircle, XCircle, AlertCircle, Clock, AlertTriangle } from 'lucide-react';
import clsx from 'clsx';

type Verdict = 'REAL' | 'FAKE' | 'UNCERTAIN';
type Status = 'pending' | 'processing' | 'completed' | 'failed';

interface VerdictCardProps {
  verdict?: Verdict | null;
  confidence?: number | null;
  status?: Status;
  showDisclaimer?: boolean;
}

const verdictConfig = {
  REAL: {
    icon: CheckCircle,
    label: 'Likely Real',
    bgColor: 'bg-success-50',
    borderColor: 'border-success-200',
    textColor: 'text-success-700',
    iconColor: 'text-success-500',
  },
  FAKE: {
    icon: XCircle,
    label: 'Likely Fake',
    bgColor: 'bg-danger-50',
    borderColor: 'border-danger-200',
    textColor: 'text-danger-700',
    iconColor: 'text-danger-500',
  },
  UNCERTAIN: {
    icon: AlertCircle,
    label: 'Uncertain',
    bgColor: 'bg-amber-50',
    borderColor: 'border-amber-200',
    textColor: 'text-amber-700',
    iconColor: 'text-amber-500',
  },
};

export default function VerdictCard({
  verdict,
  confidence,
  status = 'completed',
  showDisclaimer = true,
}: VerdictCardProps) {
  // Handle non-completed statuses
  if (status === 'pending' || status === 'processing') {
    return (
      <div className="card border-primary-200 bg-primary-50">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-primary-100 rounded-full">
            <Clock className="w-8 h-8 text-primary-500 animate-pulse" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-primary-700">
              {status === 'pending' ? 'Queued for Processing' : 'Analyzing...'}
            </h3>
            <p className="text-sm text-primary-600">
              {status === 'pending'
                ? 'Your file is in the queue'
                : 'Processing your media file'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (status === 'failed') {
    return (
      <div className="card border-danger-200 bg-danger-50">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-danger-100 rounded-full">
            <XCircle className="w-8 h-8 text-danger-500" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-danger-700">Analysis Failed</h3>
            <p className="text-sm text-danger-600">
              An error occurred during processing
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Completed with verdict
  if (!verdict) {
    return null;
  }

  const config = verdictConfig[verdict];
  const Icon = config.icon;

  return (
    <div className={clsx('card border', config.borderColor, config.bgColor)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className={clsx('p-3 rounded-full', `${config.bgColor}`)}>
            <Icon className={clsx('w-10 h-10', config.iconColor)} />
          </div>
          <div>
            <h3 className={clsx('text-2xl font-bold', config.textColor)}>
              {config.label}
            </h3>
            {confidence !== null && confidence !== undefined && (
              <p className="text-sm text-gray-600 mt-1">
                Confidence: {(confidence * 100).toFixed(1)}%
              </p>
            )}
          </div>
        </div>
        {confidence !== null && confidence !== undefined && (
          <div className="text-right">
            <div className="text-4xl font-bold text-gray-900">
              {(confidence * 100).toFixed(0)}%
            </div>
            <div className="text-sm text-gray-500">confidence</div>
          </div>
        )}
      </div>

      {showDisclaimer && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-start gap-2 text-sm text-gray-500">
            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <p>
              This is a forensic estimate, not certainty. Results should be verified by
              qualified experts before making any decisions.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
