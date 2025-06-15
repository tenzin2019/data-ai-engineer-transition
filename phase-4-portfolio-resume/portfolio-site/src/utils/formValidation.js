import DOMPurify from 'dompurify';

// Form validation and sanitization
export const validateEmail = (email) => {
  const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  return emailRegex.test(email);
};

export const sanitizeInput = (input) => {
  // Remove any HTML tags
  const sanitized = input.replace(/<[^>]*>/g, '');
  // Remove any script tags
  return sanitized.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
};

export const validateForm = (formData) => {
  const errors = {};
  const sanitizedData = {};

  // Validate and sanitize name
  if (!formData.name.trim()) {
    errors.name = 'Name is required';
  } else if (formData.name.length > 100) {
    errors.name = 'Name must be less than 100 characters';
  } else {
    sanitizedData.name = DOMPurify.sanitize(formData.name.trim());
  }

  // Validate and sanitize email
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!formData.email.trim()) {
    errors.email = 'Email is required';
  } else if (!emailRegex.test(formData.email)) {
    errors.email = 'Please enter a valid email address';
  } else {
    sanitizedData.email = DOMPurify.sanitize(formData.email.trim().toLowerCase());
  }

  // Validate and sanitize message
  if (!formData.message.trim()) {
    errors.message = 'Message is required';
  } else if (formData.message.length > 1000) {
    errors.message = 'Message must be less than 1000 characters';
  } else {
    sanitizedData.message = DOMPurify.sanitize(formData.message.trim());
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors,
    sanitizedData
  };
}; 