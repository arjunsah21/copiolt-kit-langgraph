import { useState, useEffect } from 'react';
import { CopilotKit } from '@copilotkit/react-core';
import { CopilotChat } from '@copilotkit/react-ui';
import '@copilotkit/react-ui/styles.css';
import './App.css';

function App() {
  const [location, setLocation] = useState<{ latitude: number; longitude: number } | null>(null);
  const [locationError, setLocationError] = useState<string | null>(null);
  const [isLoadingLocation, setIsLoadingLocation] = useState(true);
  const [backendReady, setBackendReady] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);

  // Check if backend is ready
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        if (response.ok) {
          setBackendReady(true);
          console.log('Backend is ready');
        } else {
          setBackendError('Backend is not responding correctly');
        }
      } catch (error) {
        console.error('Backend health check failed:', error);
        setBackendError('Cannot connect to backend. Make sure it\'s running on port 8000.');
      }
    };

    checkBackend();
    // Retry every 3 seconds if backend is not ready
    const interval = setInterval(() => {
      if (!backendReady) {
        checkBackend();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [backendReady]);

  // Request geolocation on component mount
  useEffect(() => {
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const loc = {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
          };
          setLocation(loc);
          setIsLoadingLocation(false);
          console.log('Location obtained:', loc.latitude, loc.longitude);
        },
        (error) => {
          console.error('Geolocation error:', error);
          setLocationError(error.message);
          setIsLoadingLocation(false);
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 0
        }
      );
    } else {
      setLocationError('Geolocation is not supported by your browser');
      setIsLoadingLocation(false);
    }
  }, []);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🗺️ Local Guide</h1>
        <p>Ask me about weather or restaurants near you!</p>
      </header>

      {/* Backend Status */}
      {!backendReady && (
        <div className="location-status error">
          <p>⚠️ Backend Status: {backendError || 'Connecting...'}</p>
          <p style={{ fontSize: '0.85rem' }}>
            Make sure to run: <code>cd backend && uv run uvicorn src.main:app --reload</code>
          </p>
        </div>
      )}

      {/* Location Status */}
      {isLoadingLocation && (
        <div className="location-status loading">
          <p>📍 Getting your location...</p>
        </div>
      )}

      {locationError && (
        <div className="location-status error">
          <p>⚠️ Location Error: {locationError}</p>
          <p>Please enable location permissions to use this app.</p>
        </div>
      )}

      {location && !isLoadingLocation && backendReady && (
        <div className="location-status success">
          <p>✅ Ready! Location: {location.latitude.toFixed(4)}, {location.longitude.toFixed(4)}</p>
        </div>
      )}

      <div className="chat-container">
        {!backendReady ? (
          <div className="placeholder">
            <h2>🔌 Connecting to Backend...</h2>
            <p>Waiting for the backend server to start.</p>
            <p style={{ fontSize: '0.9rem', marginTop: '1rem', color: '#666' }}>
              {backendError || 'Checking http://localhost:8000/health'}
            </p>
          </div>
        ) : !location && !isLoadingLocation ? (
          <div className="placeholder">
            <h2>📍 Location Required</h2>
            <p>Please enable location access to use the Local Guide.</p>
            <p style={{ fontSize: '0.9rem', marginTop: '1rem', color: '#666' }}>
              Click the location icon in your browser's address bar to enable location access.
            </p>
          </div>
        ) : location && backendReady ? (
          <CopilotKit
            runtimeUrl="http://localhost:8000/api/copilotkit"
            properties={{
              location: {
                latitude: location.latitude,
                longitude: location.longitude,
              },
            }}
            showDevConsole={false}
          >
            <CopilotChat
              instructions="You are a helpful local guide assistant. Help users with weather information and restaurant recommendations based on their location."
              labels={{
                title: 'Local Guide Assistant',
                initial: 'Hi! I can help you with weather or find restaurants nearby. What would you like to know?',
              }}
              suggestions={[
                { title: "What's the weather like?", message: "What's the current weather?" },
                { title: 'Find restaurants', message: 'Show me restaurants near me' },
                { title: 'Temperature', message: "What's the temperature right now?" },
                { title: 'Food options', message: 'What are some good places to eat nearby?' },
              ]}
            />
          </CopilotKit>
        ) : (
          <div className="placeholder">
            <p>Loading...</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
