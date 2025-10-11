import React from 'react';
import { AlertTriangle } from 'lucide-react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white flex items-center justify-center p-4">
          <div className="bg-red-900/50 border-2 border-red-500 rounded-xl p-8 max-w-lg">
            <div className="flex items-center gap-4 mb-4">
              <AlertTriangle size={48} className="text-red-400" />
              <h2 className="text-2xl font-bold">Something went wrong</h2>
            </div>
            <p className="text-slate-300 mb-6">
              An unexpected error occurred. Please refresh the page and try again.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg transition font-medium"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
