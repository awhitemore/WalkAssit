import './App.css'

const playAudio = async (voice, text) => {
  const voiceIds = {
    'john': 'JBFqnCBsd6RMkjVDRZzb',
    'rachel': '21m00Tcm4TlvDq8ikWAM',
    'ai-adam': 'pNInz6obpgDQGcFmaJgB',
    'alice': 'Xb7hH8MSUJpSbSDYk0k2',
    'charlie': 'IKne3meq5aSn9XLyUdCD'
  }
  const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceIds[voice]}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'xi-api-key': import.meta.env.VITE_ELEVENLABS_API_KEY,
    },
    body: JSON.stringify({
      text: text,
      model_id: 'eleven_multilingual_v2',
    }),
  });

  const audioBlob = await response.blob();
  const audioUrl = URL.createObjectURL(audioBlob);
  const audio = new Audio(audioUrl);
  audio.play();
};
function App() {

  return (
    <>
      <img src="/walk-assist.png" alt="Walk Assist" height={100} />
      <h1>Walk Assist</h1>
      <div className="card">
        <button onClick={() => playAudio('charlie', 'Hello, how are you doing today?')}>
          Play Audio
        </button>
      </div>
    </>
  )
}

export default App
