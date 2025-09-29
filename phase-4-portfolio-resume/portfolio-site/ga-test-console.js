/**
 * Google Analytics Test Script for Browser Console
 * 
 * Instructions:
 * 1. Open your portfolio site in browser (http://localhost:XXXX)
 * 2. Open Developer Tools (F12)
 * 3. Go to Console tab
 * 4. Copy and paste this entire script
 * 5. Press Enter to run
 */

console.log('🔍 Google Analytics Test Starting...');
console.log('=====================================');

// Test 1: Check if GA variables are available
console.log('\n1️⃣ Checking GA Configuration...');
try {
  // These should be available from the analytics module
  console.log('✓ window.gtag available:', typeof window.gtag !== 'undefined');
  console.log('✓ window.dataLayer available:', Array.isArray(window.dataLayer));
  console.log('✓ dataLayer length:', window.dataLayer?.length || 0);
} catch (error) {
  console.error('✗ Error checking GA variables:', error.message);
}

// Test 2: Check environment variables (if accessible)
console.log('\n2️⃣ Checking Environment...');
console.log('✓ Current URL:', window.location.href);
console.log('✓ User Agent:', navigator.userAgent.substring(0, 50) + '...');

// Test 3: Check for GA script loading
console.log('\n3️⃣ Checking GA Script Loading...');
const gaScripts = document.querySelectorAll('script[src*="googletagmanager"]');
console.log('✓ GA scripts found:', gaScripts.length);
gaScripts.forEach((script, index) => {
  console.log(`  Script ${index + 1}:`, script.src);
});

// Test 4: Check dataLayer contents
console.log('\n4️⃣ Checking DataLayer Contents...');
if (window.dataLayer && window.dataLayer.length > 0) {
  console.log('✓ DataLayer entries:');
  window.dataLayer.forEach((entry, index) => {
    console.log(`  Entry ${index + 1}:`, entry);
  });
} else {
  console.log('⚠️  DataLayer is empty or not initialized');
}

// Test 5: Test manual event tracking
console.log('\n5️⃣ Testing Manual Event Tracking...');
if (typeof window.gtag === 'function') {
  console.log('✓ Sending test event...');
  window.gtag('event', 'test_event', {
    event_category: 'manual_test',
    event_label: 'console_test',
    custom_parameter: 'browser_console_test'
  });
  console.log('✓ Test event sent! Check Network tab for requests to analytics.');
} else {
  console.log('✗ gtag function not available - GA not properly initialized');
}

// Test 6: Network requests check instruction
console.log('\n6️⃣ Network Verification...');
console.log('📋 To verify GA is working:');
console.log('   1. Go to Network tab in DevTools');
console.log('   2. Filter by "analytics" or "google"');
console.log('   3. Look for requests to google-analytics.com');
console.log('   4. Refresh page and interact with site');

// Test 7: Real-time testing instructions
console.log('\n7️⃣ Real-time Testing...');
console.log('📊 To see real-time data:');
console.log('   1. Go to https://analytics.google.com/');
console.log('   2. Navigate to Realtime → Overview');
console.log('   3. Should see your current session');
console.log('   4. Interact with portfolio to see events');

console.log('\n✅ Google Analytics Test Complete!');
console.log('=====================================');

// Return useful debugging info
const debugInfo = {
  gtagAvailable: typeof window.gtag !== 'undefined',
  dataLayerAvailable: Array.isArray(window.dataLayer),
  dataLayerLength: window.dataLayer?.length || 0,
  gaScriptsFound: document.querySelectorAll('script[src*="googletagmanager"]').length,
  currentURL: window.location.href,
  timestamp: new Date().toISOString()
};

console.log('\n📋 Debug Summary:', debugInfo);
return debugInfo;
