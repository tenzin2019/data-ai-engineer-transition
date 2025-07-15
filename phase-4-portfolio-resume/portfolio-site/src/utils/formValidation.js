import DOMPurify from 'dompurify';

// Stronger email validation
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

// Basic anti-spam: block URLs, common spammy keywords, and check honeypot
export const isSpam = (formData) => {
  // Honeypot field (should be empty)
  if (formData.honeypot && formData.honeypot.trim() !== '') return true;
  // Block URLs in message
  if (/https?:\/\//i.test(formData.message)) return true;
  // Block common spammy keywords
  const spamKeywords = [
    'viagra', 'free money', 'bitcoin', 'loan', 'casino', 'porn', 'sex', 'escort', 'win big', 'click here', 'buy now', 'credit card', 'investment', 'urgent', 'offer', 'prize', 'guaranteed', 'cheap', 'work from home', 'weight loss', 'miracle', 'earn $', 'make money', 'crypto', 'telegram', 'whatsapp', 'skype', 'call now', 'sms', 'sms to', 'sms:', 'whatsapp:', 'telegram:'
  ];
  const msg = formData.message.toLowerCase();
  if (spamKeywords.some(word => msg.includes(word))) return true;
  return false;
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

  // Validate and sanitize email (stronger check)
  if (!formData.email.trim()) {
    errors.email = 'Email is required';
  } else if (!validateEmail(formData.email)) {
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

  // Honeypot (should be empty)
  sanitizedData.honeypot = formData.honeypot || '';

  return {
    isValid: Object.keys(errors).length === 0,
    errors,
    sanitizedData
  };
}; 