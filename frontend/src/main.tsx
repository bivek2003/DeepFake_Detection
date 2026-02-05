import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './styles/index.css';

import Dashboard from './pages/Dashboard';
import UploadImage from './pages/UploadImage';
import UploadVideo from './pages/UploadVideo';
import ResultDetail from './pages/ResultDetail';
import Layout from './components/Layout';

import LandingPage from './pages/LandingPage';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route element={<Layout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/upload/image" element={<UploadImage />} />
          <Route path="/upload/video" element={<UploadVideo />} />
          <Route path="/result/:id" element={<ResultDetail />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
