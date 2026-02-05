import { useEffect, useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { Shield, Upload, Video, Home, Info } from 'lucide-react';
import DemoBanner from './DemoBanner';
import { getModelInfo, ModelInfo } from '../api';

const navItems = [
  { path: '/dashboard', label: 'Dashboard', icon: Home },
  { path: '/upload/image', label: 'Image Analysis', icon: Upload },
  { path: '/upload/video', label: 'Video Analysis', icon: Video },
];

export default function Layout() {
  const location = useLocation();
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    getModelInfo()
      .then(setModelInfo)
      .catch(console.error);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Demo Mode Banner */}
      {modelInfo?.demo_mode && <DemoBanner />}

      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link to="/dashboard" className="flex items-center gap-3">
              <div className="p-2 bg-primary-600 rounded-lg">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-900">Deepfake Detection</h1>
                <p className="text-xs text-gray-500">Defensive Media Forensics</p>
              </div>
            </Link>

            {/* Navigation */}
            <nav className="hidden md:flex items-center gap-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`px-4 py-2 rounded-lg flex items-center gap-2 text-sm font-medium transition-colors ${isActive
                      ? 'bg-primary-50 text-primary-700'
                      : 'text-gray-600 hover:bg-gray-100'
                      }`}
                  >
                    <Icon className="w-4 h-4" />
                    {item.label}
                  </Link>
                );
              })}
            </nav>

            {/* Model Info */}
            {modelInfo && (
              <div className="hidden lg:flex items-center gap-2 text-sm text-gray-500">
                <Info className="w-4 h-4" />
                <span>v{modelInfo.model_version}</span>
                <span className="text-gray-300">|</span>
                <span>{modelInfo.device}</span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <p className="text-sm text-gray-500">
              © 2024 Deepfake Detection Platform. Defensive forensics only.
            </p>
            <p className="text-xs text-gray-400 max-w-xl text-center md:text-right">
              ⚠️ This is a forensic estimate, not certainty. Results should be verified by
              qualified experts before making any decisions.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
