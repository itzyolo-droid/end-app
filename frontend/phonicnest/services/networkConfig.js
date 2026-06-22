// Network Configuration Helper
// This file helps manage different API URLs for different environments

const ENVIRONMENTS = {
  DEVELOPMENT: 'development',
  PRODUCTION: 'production',
  LOCAL_NETWORK: 'local_network',
};

// Try to auto-detect your computer's LAN IP from Expo, fallback to manual IP
const getLocalIP = () => {
  try {
    // Lazy require to avoid hard dependency in non-Expo contexts
    const Constants = require('expo-constants').default;
    const hostUri = (Constants?.expoConfig?.hostUri) || (Constants?.manifest?.hostUri) || '';
    // Examples: "192.168.1.50:19000", "10.7.40.174:19000"
    const match = /^(\d+\.\d+\.\d+\.\d+):\d+$/.exec(hostUri || '');
    if (match && match[1]) {
      return match[1];
    }
  } catch (e) {
    // Ignore and use manual fallback
  }
  // Manual fallback: replace with your actual IPv4 from 'ipconfig'
  return '192.168.0.121';
};

// Configuration for different environments
const config = {
  [ENVIRONMENTS.DEVELOPMENT]: {
    API_BASE_URL: 'http://localhost:8000',
    TIMEOUT: 30000,
  },
  [ENVIRONMENTS.LOCAL_NETWORK]: {
    API_BASE_URL: `http://${getLocalIP()}:8000`,
    TIMEOUT: 30000,
  },
  [ENVIRONMENTS.PRODUCTION]: {
    API_BASE_URL: 'https://your-production-domain.com', // Replace with your production URL
    TIMEOUT: 30000,
  },
};

// Current environment - change this as needed
const CURRENT_ENVIRONMENT = ENVIRONMENTS.LOCAL_NETWORK;

// Export the current configuration with dynamic fallback to EXPO_PUBLIC_API_URL
const selectedConfig = config[CURRENT_ENVIRONMENT] || config[ENVIRONMENTS.DEVELOPMENT];
export const currentConfig = {
  ...selectedConfig,
  API_BASE_URL: process.env.EXPO_PUBLIC_API_URL || selectedConfig.API_BASE_URL,
};


// Helper function to get configuration for a specific environment
export const getConfig = (environment = CURRENT_ENVIRONMENT) => {
  return config[environment] || config[ENVIRONMENTS.DEVELOPMENT];
};

// Helper function to switch environment
export const switchEnvironment = (environment) => {
  if (config[environment]) {
    console.log(`Switching to ${environment} environment`);
    return config[environment];
  }
  console.warn(`Environment ${environment} not found, using development`);
  return config[ENVIRONMENTS.DEVELOPMENT];
};

export default currentConfig;
