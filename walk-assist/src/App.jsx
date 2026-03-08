import './App.css'
import playAudio from './audio'
import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'


function App() {
  const [started, setStarted] = useState(false);
  const navigate = useNavigate();

  const handleStart = () => {
    if (started) return;
    setStarted(true);
    playAudio('charlie', 'Welcome to Walk Assist! I am Charlie, your virtual walking companion. I will help you stay safe as you navigate the world. Object detection is enabled. I will help you avoid obstacles and stay on the path. Good luck!');
  };

  useEffect(() => {
    if (started) {
      const timer = setTimeout(() => {
        navigate('/depth');
      }, 17000);
      return () => clearTimeout(timer);
    }
  }, [started, navigate]);

  return (
    <div onClick={handleStart} style={{ minHeight: '100vh', cursor: started ? 'default' : 'pointer' }}>
      <img src="/walk-assist.png" alt="Walk Assist" height={100} />
      <h1>Walk Assist</h1>
      {!started && <p style={{ opacity: 0.6 }}>Tap anywhere to start</p>}
    </div>
  )
}

export default App
