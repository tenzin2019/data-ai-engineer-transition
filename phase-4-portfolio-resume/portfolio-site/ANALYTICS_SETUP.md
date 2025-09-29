# Google Analytics Setup Guide

This portfolio site is configured with Google Analytics 4 (GA4) for comprehensive tracking and analytics.

## 🚀 Quick Setup

### 1. Get Your Google Analytics Measurement ID

1. Go to [Google Analytics](https://analytics.google.com/)
2. Create a new GA4 property or use an existing one
3. Get your Measurement ID (format: `G-XXXXXXXXXX`)

### 2. Configure Environment Variable

Create a `.env` file in the project root:

```bash
# .env
VITE_GA_MEASUREMENT_ID=G-2C336M7K6E
```

Replace `G-XXXXXXXXXX` with your actual Measurement ID.

### 3. Build and Deploy

```bash
npm run build
```

The site will automatically initialize Google Analytics when deployed.

## 📊 What's Being Tracked

### Automatic Tracking
- **Page Views**: Initial portfolio load
- **Enhanced Measurements**: Scroll depth, outbound clicks, file downloads
- **User Engagement**: Time on page, bounce rate

### Custom Events
- **Contact Form**: Submission attempts, successes, and errors
- **Project Views**: Clicks on project links and demo buttons
- **Skills Interaction**: Views of the skills section
- **Navigation**: Section scrolling and interactions

### Event Categories
- `portfolio` - General portfolio interactions
- `projects` - Project-related actions
- `contact` - Contact form interactions
- `skills` - Skills section engagement
- `downloads` - File downloads (if implemented)

## 🔒 Privacy & Compliance

### Privacy Features Enabled
- **IP Anonymization**: User IP addresses are anonymized
- **Cookie Settings**: Secure and SameSite=Strict cookies
- **Enhanced Measurement**: Automatic event tracking enabled

### GDPR/Privacy Considerations
- Analytics only loads in production environments
- No personal data is collected beyond standard web analytics
- Users can opt out using browser settings or ad blockers

## 🛠 Development vs Production

### Development Environment
- Analytics **disabled** by default in development
- No tracking scripts loaded during `npm run dev`
- Console warnings if measurement ID is not set

### Production Environment
- Analytics **enabled** automatically
- Full event tracking active
- Real-time data available in GA4 dashboard

## 📈 Analytics Dashboard

Once configured, you can view analytics data at:
- [Google Analytics Dashboard](https://analytics.google.com/)
- Real-time reports available within minutes
- Historical data builds over time

### Key Metrics to Monitor
- **Page Views**: Total visits to your portfolio
- **User Engagement**: How long visitors stay
- **Contact Form**: Conversion rates and form completion
- **Project Interest**: Which projects get the most attention
- **Traffic Sources**: Where visitors come from

## 🔧 Customization

### Adding New Events

To track additional interactions, use the provided utility functions:

```javascript
import { trackEvent, trackPortfolioEvent } from '../utils/analytics';

// Generic event tracking
trackEvent('custom_event', {
  category: 'engagement',
  label: 'specific_action',
  value: 1
});

// Portfolio-specific event
trackPortfolioEvent('button_click', 'hero_section');
```

### Event Parameters

All tracking functions accept optional parameters:
- `category`: Event category (default: 'engagement')
- `label`: Event label for more specific tracking
- `value`: Numeric value for the event
- Custom parameters as needed

## 🚨 Troubleshooting

### Analytics Not Working?

1. **Check Environment Variable**
   ```bash
   echo $VITE_GA_MEASUREMENT_ID
   ```

2. **Verify Measurement ID Format**
   - Should start with `G-`
   - Followed by 10 alphanumeric characters
   - Example: `G-1234567890`

3. **Check Browser Console**
   - Look for GA-related errors
   - Ensure no ad blockers are interfering

4. **Verify in GA4 Dashboard**
   - Check Real-time reports
   - May take 24-48 hours for full data

### Common Issues

- **No Data in GA4**: Ensure measurement ID is correct and site is live
- **Events Not Firing**: Check browser console for JavaScript errors
- **Ad Blockers**: Some users may have analytics blocked (this is normal)

## 📝 Notes

- This setup uses Google Analytics 4 (GA4), not Universal Analytics
- All tracking is privacy-compliant with modern web standards
- Analytics gracefully degrades if blocked or disabled
- No impact on site performance or user experience

For support or questions about the analytics implementation, refer to the portfolio documentation or contact the developer.
