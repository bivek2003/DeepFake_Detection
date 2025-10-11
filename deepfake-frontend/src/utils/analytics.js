export const logEvent = (eventName, data = {}) => {
  if (import.meta.env.DEV) {
    console.log('📊 Analytics Event:', eventName, data);
  }
};

export const logPrediction = (type, result, duration) => {
  logEvent('prediction_complete', {
    type,
    prediction: result.prediction,
    confidence: result.confidence,
    duration_ms: duration,
  });
};

export const logError = (error, context = {}) => {
  console.error('❌ Error:', error, context);
  logEvent('error_occurred', {
    message: error.message,
    ...context,
  });
};

export const logPageView = (page) => {
  logEvent('page_view', { page });
};

export const logModeChange = (mode) => {
  logEvent('mode_change', { mode });
};
