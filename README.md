# WalkAssist

A real-time object detection and audio notification system designed to assist visually impaired users while walking. The system uses computer vision to detect objects and depth estimation to determine distances, then announces detected objects via text-to-speech.

## Features

- **Real-time Object Detection**: Uses YOLO26 nano segmentation model to detect urban environment objects (people, cars, bicycles, traffic lights, etc.)
- **Depth Estimation**: MiDaS depth estimation provides pseudo-distance measurements to detected objects
- **Audio Announcements**: ElevenLabs text-to-speech API announces new objects as they appear
- **Web Dashboard**: React frontend displays detection logs in real-time

## Tech Stack

### Backend / Computer Vision
- **Python 3.x** - Core runtime
- **PyTorch** - Deep learning framework for model inference
- **OpenCV** - Camera capture and image processing
- **Ultralytics YOLO** - Object detection and segmentation
- **MiDaS (Intel ISL)** - Monocular depth estimation
- **Flask** - Lightweight web server for log API

### Frontend
- **React 19** - UI framework
- **Vite** - Build tool and dev server
- **ElevenLabs API** - Text-to-speech audio generation

## Architecture

The system runs as three separate processes:

```
┌─────────────────┐     POST /logs     ┌─────────────────┐     GET /logs      ┌─────────────────┐
│   bothModel.py  │ ─────────────────► │    server.py    │ ◄───────────────── │  React Frontend │
│  (Detection)    │                    │   (Log Server)  │                    │   (Dashboard)   │
└─────────────────┘                    └─────────────────┘                    └─────────────────┘
        │                                                                              │
        ▼                                                                              ▼
   Camera Feed                                                                  Audio Output
   + CV Window                                                              (ElevenLabs TTS)
```

## Running the Application

**All three processes must run simultaneously in separate terminals.**

### Terminal 1: Start the Log Server
```bash
cd /path/to/WalkAssist
python server.py
```
The server runs on `http://localhost:5001`

### Terminal 2: Start the Detection Script
```bash
cd /path/to/WalkAssist
./models_script.sh
```
Or manually:
```bash
cd /path/to/WalkAssist
.venv/bin/python3 bothModel.py
```

### Terminal 3: Start the React Frontend
```bash
cd /path/to/WalkAssist/walk-assist
npm run dev
```
The frontend runs on `http://localhost:5173`

### First-Time Setup

1. Create a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Install frontend dependencies:
   ```bash
   cd walk-assist
   npm install
   ```

3. Create `.env` file in `walk-assist/` with your ElevenLabs API key:
   ```
   VITE_ELEVENLABS_API_KEY=your_api_key_here
   ```

4. Grant camera permissions to Terminal in **System Settings → Privacy & Security → Camera**

## Server API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/logs` | GET | Retrieve all detection logs |
| `/logs` | POST | Add a new log entry |
| `/logs/clear` | GET/POST | Clear all logs |

### Log Entry Format
```json
{
  "timestamp": "2026-03-07 17:30:45",
  "type": "detection",
  "message": "Detected person at 125.3 units"
}
```

## Audio System

The frontend uses the ElevenLabs API to generate text-to-speech audio:
- Audio plays when new object types are detected
- 2-second cooldown between announcements to prevent spam
- Announcements reset when logs are cleared (every 5 seconds)

**Note:** Due to browser autoplay restrictions, the user must tap/click the screen before audio can play.

## Known Issues & Future Improvements

### Frame Streaming Latency

We initially attempted to stream video frames from the Python backend to the React frontend via MJPEG streaming. This approach had several issues:

1. **High Latency**: The overhead of encoding frames to JPEG, transmitting over HTTP, and decoding in the browser introduced significant delays (200-500ms+)
2. **Resource Intensive**: Running both the CV processing and frame encoding/streaming in Python consumed excessive CPU
3. **Browser Compatibility**: MJPEG streams can be unreliable across different browsers

**Current Solution**: We decoupled the systems - the Python script displays frames locally via OpenCV while only sending lightweight text logs to the server. The React frontend displays logs without video, avoiding latency issues entirely.

### Object Re-Announcement

The current system tracks which object *types* have been announced (e.g., "person", "car") and avoids re-announcing the same type. However, this creates issues:

1. **Same Person, Multiple Announcements**: If a person briefly leaves and re-enters the frame, they may be announced again
2. **New Person Not Announced**: If one person is in frame and another enters, the second person won't be announced since "person" was already seen
3. **Distance Changes Ignored**: Objects moving closer/further aren't re-announced

**Future Improvements**:
- Implement object tracking (e.g., DeepSORT) to assign unique IDs to individuals
- Announce based on object ID rather than type
- Add proximity alerts when objects cross distance thresholds
- Use spatial awareness to announce position (left, right, ahead)

## Project Structure

```
WalkAssist/
├── bothModel.py          # Main detection script (YOLO + MiDaS)
├── server.py             # Flask log server
├── models_script.sh      # Shell script to run detection
├── requirements.txt      # Python dependencies
├── yolo26n-seg.pt        # YOLO model weights
└── walk-assist/          # React frontend
    ├── src/
    │   ├── App.jsx       # Main app with welcome screen
    │   ├── DepthView.jsx # Detection dashboard
    │   ├── audio.jsx     # ElevenLabs TTS wrapper
    │   └── *.css         # Styles
    ├── package.json
    └── .env              # API keys (not committed)
```

## License

MIT
