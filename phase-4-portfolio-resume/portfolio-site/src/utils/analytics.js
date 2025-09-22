// Google Analytics 4 Configuration
export const GA_MEASUREMENT_ID = import.meta.env.VITE_GA_MEASUREMENT_ID || 'G-2C336M7K6E';

// Initialize Google Analytics
export const initGA = () => {
  // Only initialize when measurement ID is available and valid
  if (typeof window === 'undefined' || !GA_MEASUREMENT_ID || GA_MEASUREMENT_ID === 'G-XXXXXXXXXX') {
    console.log('Google Analytics not initialized:', {
      hasWindow: typeof window !== 'undefined',
      measurementId: GA_MEASUREMENT_ID,
      environment: import.meta.env.MODE
    });
    return;
  }

  console.log('Initializing Google Analytics with ID:', GA_MEASUREMENT_ID);

  // Load gtag script
  const script1 = document.createElement('script');
  script1.async = true;
  script1.src = `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`;
  document.head.appendChild(script1);

  // Initialize gtag
  window.dataLayer = window.dataLayer || [];
  function gtag() {
    window.dataLayer.push(arguments);
  }
  window.gtag = gtag;

  gtag('js', new Date());
  gtag('config', GA_MEASUREMENT_ID, {
    // Enhanced measurement for better tracking
    enhanced_measurement: true,
    // Privacy-friendly settings
    anonymize_ip: true,
    // Cookie settings
    cookie_flags: 'SameSite=Strict;Secure',
  });
};

// Track page views
export const trackPageView = (url, title) => {
  if (typeof window === 'undefined' || !window.gtag) {
    return;
  }

  window.gtag('config', GA_MEASUREMENT_ID, {
    page_title: title,
    page_location: url,
  });
};

// Track custom events
export const trackEvent = (eventName, parameters = {}) => {
  if (typeof window === 'undefined' || !window.gtag) {
    return;
  }

  window.gtag('event', eventName, {
    event_category: parameters.category || 'engagement',
    event_label: parameters.label || '',
    value: parameters.value || 1,
    ...parameters,
  });
};

// Track specific portfolio events
export const trackPortfolioEvent = (action, section = '') => {
  trackEvent('portfolio_interaction', {
    category: 'portfolio',
    label: section,
    action: action,
  });
};

// Track project views
export const trackProjectView = (projectName) => {
  trackEvent('project_view', {
    category: 'projects',
    label: projectName,
  });
};

// Track contact form interactions
export const trackContactEvent = (action) => {
  trackEvent('contact_interaction', {
    category: 'contact',
    label: action,
  });
};

// Track skill section interactions
export const trackSkillsInteraction = () => {
  trackEvent('skills_view', {
    category: 'skills',
    label: 'skills_randomization',
  });
};

// Track resume/CV downloads
export const trackDownload = (fileName) => {
  trackEvent('file_download', {
    category: 'downloads',
    label: fileName,
  });
};
