# Deepfake Detector - Frontend (Vite + React)

Modern web interface for deepfake detection system built with Vite, React, and Tailwind CSS.

## ✨ Features

- 📸 **Webcam Capture**: Real-time face detection and analysis
- 📤 **File Upload**: Support for images, videos, and audio files
- 🎯 **Real-time Results**: Instant feedback with confidence scores
- 🎨 **Modern UI**: Beautiful gradient design with Tailwind CSS
- ⌨️ **Keyboard Shortcuts**: Quick navigation and controls
- 📱 **Responsive**: Works on desktop and mobile browsers
- 🎭 **Error Handling**: Graceful error boundaries and user feedback

## 🚀 Prerequisites

- Node.js 20+ and npm
- Backend API running (default: http://localhost:8000)

## 📦 Installation

```bash
npm install
```

## 🏃 Development

```bash
npm run dev
```

Open http://localhost:5173

## 🏗️ Build for Production

```bash
npm run build
npm run preview
```

## ⌨️ Keyboard Shortcuts

- `U` - Switch to Upload mode
- `W` - Switch to Webcam mode
- `R` - Reset/Analyze another
- `Esc` - Stop webcam (when active)

## 📁 Project Structure

```
src/
├── components/       # React components
├── hooks/           # Custom React hooks
├── services/        # API services
├── utils/           # Utility functions
├── constants/       # App constants
├── App.jsx          # Main app
└── main.jsx         # Entry point
```

## 🔧 Environment Variables

- `VITE_API_URL` - Backend API endpoint (default: http://localhost:8000)

## 🛠️ Technologies

- Vite - Next generation frontend tooling
- React 18 - UI library
- Tailwind CSS - Utility-first CSS
- Axios - HTTP client
- Lucide React - Icon library

## 📄 License

Part of the Deepfake Detection project - Phase 4 Frontend
