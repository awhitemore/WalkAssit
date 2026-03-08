import './DepthView.css'
import { useState, useEffect, useRef } from 'react'
import playAudio from './audio'

function DepthView() {
    const [started, setStarted] = useState(false);
    const [logs, setLogs] = useState([]);
    const seenObjects = useRef(new Set());
    const lastAnnouncedTime = useRef(0);

    const handleStart = () => {
        if (started) return;
        setStarted(true);
    };

    // Extract object type from log message like "Detected person at 125.3 units"
    const extractObjectType = (message) => {
        const match = message.match(/Detected (\w+) at/);
        return match ? match[1] : null;
    };

    // Format timestamp to show just time in 12-hour format
    const formatTime = (timestamp) => {
        // timestamp is "YYYY-MM-DD HH:MM:SS"
        const timePart = timestamp.split(' ')[1];
        if (!timePart) return timestamp;

        const [hours, minutes, seconds] = timePart.split(':');
        const hour = parseInt(hours, 10);
        const ampm = hour >= 12 ? 'PM' : 'AM';
        const hour12 = hour % 12 || 12;

        return `${hour12}:${minutes}:${seconds} ${ampm}`;
    };

    useEffect(() => {
        if (!started) return;

        const fetchLogs = async () => {
            try {
                const response = await fetch('http://localhost:5001/logs');
                const data = await response.json();

                // Check for new object types
                const currentTime = Date.now();
                const newObjects = [];

                data.forEach(log => {
                    if (log.type === 'detection') {
                        const objectType = extractObjectType(log.message);
                        if (objectType && !seenObjects.current.has(objectType)) {
                            seenObjects.current.add(objectType);
                            newObjects.push(objectType);
                        }
                    }
                });

                // Announce new objects (with 2 second cooldown to avoid spam)
                if (newObjects.length > 0 && currentTime - lastAnnouncedTime.current > 2000) {
                    const announcement = newObjects.length === 1
                        ? `${newObjects[0]} ahead`
                        : `${newObjects.join(' and ')} ahead`;
                    playAudio('charlie', announcement);
                    lastAnnouncedTime.current = currentTime;
                }

                setLogs(data);
            } catch (error) {
                console.error('Error fetching logs:', error);
            }
        };

        const clearLogs = async () => {
            try {
                await fetch('http://localhost:5001/logs/clear');
                // Reset seen objects when logs are cleared so they can be announced again
                seenObjects.current.clear();
            } catch (error) {
                console.error('Error clearing logs:', error);
            }
        };

        fetchLogs();
        const fetchIntervalId = setInterval(fetchLogs, 1000);
        const clearIntervalId = setInterval(clearLogs, 5000);

        return () => {
            clearInterval(fetchIntervalId);
            clearInterval(clearIntervalId);
        };
    }, [started]);

    return (
        <div className="depth-view" onClick={handleStart} style={{ cursor: started ? 'default' : 'pointer' }}>
            {!started && <p style={{ opacity: 0.6, marginBottom: '1rem' }}>Tap anywhere to continue</p>}
            <h2>Object Detection Active</h2>
            <div className="camera-grid">
                <div className="camera-feed">
                    <div className="feed-label">YOLO Segmentation</div>
                    <div className="feed-placeholder">
                        <span>📷</span>
                        <p>Processing...</p>
                    </div>
                </div>
                <div className="depth-feed">
                    <div className="feed-label">Depth Map</div>
                    <div className="depth-placeholder">
                        <div className="depth-gradient"></div>
                        <p>Processing...</p>
                    </div>
                </div>
            </div>
            <div className="logs-container">
                <div className="logs-label">Detection Logs</div>
                <div className="logs-content">
                    {logs.length === 0 ? (
                        <p className="log-empty">Waiting for detections...</p>
                    ) : (
                        logs.slice().reverse().map((log, index) => (
                            <div key={index} className={`log-entry log-${log.type}`}>
                                <span className="log-timestamp">{formatTime(log.timestamp)}</span>
                                <span className="log-message">{log.message}</span>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    )
}

export default DepthView
