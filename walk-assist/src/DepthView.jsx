import './DepthView.css'
import { useState, useEffect, useRef } from 'react'
import playAudio from './audio'

function DepthView() {
    const [started, setStarted] = useState(false);
    const [logs, setLogs] = useState([]);
    const lastAnnouncedTime = useRef(0);
    const lastAnnouncedState = useRef(''); // Track what we last announced
    const isAnnouncing = useRef(false); // Prevent overlapping Gemini calls

    const handleStart = () => {
        if (started) return;
        setStarted(true);
    };

    // Extract object type and position from log message
    const extractObjectInfo = (message) => {
        const typeMatch = message.match(/Detected (\w+)/);
        const positionMatch = message.match(/\b(left|right|middle)\b/i);
        let position = positionMatch ? positionMatch[1].toLowerCase() : null;
        // Translate "middle" to "ahead" for speech
        if (position === 'middle') position = 'ahead';
        return {
            type: typeMatch ? typeMatch[1] : null,
            position: position
        };
    };

    // Check if there's a significant change worth announcing
    // States are position arrays like ["left", "right"]
    const hasSignificantChange = (currentState, lastState) => {
        // Parse position arrays (handle empty string as empty array)
        const currentPositions = currentState ? JSON.parse(currentState) : [];
        const lastPositions = lastState ? JSON.parse(lastState) : [];

        // If both are empty, no change
        if (currentPositions.length === 0 && lastPositions.length === 0) return false;

        // If current is empty but last wasn't (or vice versa), that's a change
        if (currentPositions.length === 0 || lastPositions.length === 0) {
            return currentPositions.length !== lastPositions.length;
        }

        // Convert to sets for comparison
        const currentSet = new Set(currentPositions);
        const lastSet = new Set(lastPositions);

        // Check if any position in current is new (not in last)
        for (const pos of currentPositions) {
            if (!lastSet.has(pos)) return true;
        }

        // Check if any position from last is now missing
        for (const pos of lastPositions) {
            if (!currentSet.has(pos)) return true;
        }

        return false;
    };

    // Hardcoded exasperated alerts (no API needed)
    const exasperatedAlerts = [
        (dir, obj) => `Oh great, there's a ${obj} ${dir}. Why is everything always in the way?`,
        (dir, obj) => `Seriously? You're about to walk into a ${obj} ${dir}. Watch it!`,
        (dir, obj) => `Move! There's a ${obj} ${dir}. Can we just have one clear path?`,
        (dir, obj) => `Ugh, again? Look out for the ${obj} ${dir}. People leave stuff everywhere.`,
        (dir, obj) => `I'm begging you to avoid the ${obj} ${dir}. This is getting ridiculous.`,
        (dir, obj) => `Unbelievable. A ${obj} ${dir}. Do I really have to tell you every time?`,
        (dir, obj) => `For the love of... there's a ${obj} ${dir}. Please steer around it.`,
        (dir, obj) => `Can you listen? There's a ${obj} ${dir}. My circuits are frying.`,
        (dir, obj) => `Watch your step! A ${obj} ${dir}. Why is navigating so hard today?`,
        (dir, obj) => `Really? A ${obj} ${dir}. I'm exhausted from watching out for you.`,
        (dir, obj) => `Sigh. There's a ${obj} ${dir}. Can we find a clear path for once?`,
        (dir, obj) => `Hey, pay attention! A ${obj} ${dir}. I can't do this all day.`,
        (dir, obj) => `Great, just great. A ${obj} ${dir}. The whole world is against us.`,
        (dir, obj) => `Avoid ${dir}! There's a ${obj} there. Why is there junk everywhere?`,
        (dir, obj) => `Move it! A ${obj} ${dir}. I'm tired of saving your shins.`
    ];

    // Get exasperated announcement using hardcoded alerts
    const getExasperatedAnnouncement = (positions, types) => {
        // Format direction text
        const formatDir = (pos) => pos === 'ahead' ? 'ahead' : `on the ${pos}`;

        // Pick a random alert
        const alertFn = exasperatedAlerts[Math.floor(Math.random() * exasperatedAlerts.length)];

        if (positions.length === 1) {
            // Single detection
            const dir = formatDir(positions[0]);
            const obj = types[0] || 'obstacle';
            return alertFn(dir, obj);
        } else {
            // Multiple detections - combine them
            const descriptions = positions.map((pos, i) => {
                const obj = types[i] || types[0] || 'obstacle';
                return `${obj} ${formatDir(pos)}`;
            });
            const combined = descriptions.join(' and ');
            return `Watch out! There's a ${combined}. This is chaos!`;
        }
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

                const currentTime = Date.now();
                const timeSinceLastAnnouncement = currentTime - lastAnnouncedTime.current;

                // Group detections by timestamp (seconds) and collect positions
                const timestampGroups = new Map(); // timestamp -> Set of positions
                data.forEach(log => {
                    if (log.type === 'detection') {
                        const info = extractObjectInfo(log.message);
                        if (info.type && info.position) {
                            // Use timestamp as key (already in seconds precision)
                            if (!timestampGroups.has(log.timestamp)) {
                                timestampGroups.set(log.timestamp, { positions: new Set(), types: new Set() });
                            }
                            timestampGroups.get(log.timestamp).positions.add(info.position);
                            timestampGroups.get(log.timestamp).types.add(info.type);
                        }
                    }
                });

                // Get the most recent timestamp group's positions as the current state
                const sortedTimestamps = Array.from(timestampGroups.keys()).sort();
                const latestTimestamp = sortedTimestamps[sortedTimestamps.length - 1];
                const latestGroup = latestTimestamp ? timestampGroups.get(latestTimestamp) : null;

                // Current state is just the positions array (sorted for consistency)
                const currentPositions = latestGroup ? Array.from(latestGroup.positions).sort() : [];
                const currentTypes = latestGroup ? Array.from(latestGroup.types) : [];

                // Build objects for announcement (combine types with positions)
                const currentObjects = currentTypes.map((type, i) => ({
                    type,
                    position: currentPositions[i] || currentPositions[0] || 'ahead'
                }));

                const currentState = JSON.stringify(currentPositions);
                const significantChange = hasSignificantChange(currentState, lastAnnouncedState.current);

                console.log('Current state:', currentPositions);
                console.log('Last state:', lastAnnouncedState.current ? JSON.parse(lastAnnouncedState.current) : []);
                console.log('Significant change:', significantChange);

                // Announce only on significant change (with 5s minimum gap to avoid spam)
                const shouldAnnounce = currentPositions.length > 0 &&
                    significantChange &&
                    timeSinceLastAnnouncement >= 5000 &&
                    !isAnnouncing.current;

                if (shouldAnnounce) {
                    isAnnouncing.current = true;
                    console.log('Announcing - positions:', currentPositions, 'types:', currentTypes);
                    lastAnnouncedTime.current = currentTime;
                    lastAnnouncedState.current = currentState;

                    const announcement = getExasperatedAnnouncement(currentPositions, currentTypes);
                    console.log('Announcement:', announcement);
                    playAudio('charlie', announcement);
                    isAnnouncing.current = false;
                }

                // Accumulate logs - append new logs to existing ones (avoid duplicates by timestamp)
                setLogs(prevLogs => {
                    const existingTimestamps = new Set(prevLogs.map(l => l.timestamp + l.message));
                    const newLogs = data.filter(l => !existingTimestamps.has(l.timestamp + l.message));
                    return [...prevLogs, ...newLogs];
                });
            } catch (error) {
                console.error('Error fetching logs:', error);
            }
        };

        const clearServerLogs = async () => {
            try {
                await fetch('http://localhost:5001/logs/clear');
                // Only reset announced state if more than 3 seconds have passed since last announcement
                const timeSinceLastAnnouncement = Date.now() - lastAnnouncedTime.current;
                if (timeSinceLastAnnouncement >= 3000) {
                    lastAnnouncedState.current = '';
                }
            } catch (error) {
                console.error('Error clearing logs:', error);
            }
        };

        fetchLogs();
        const fetchIntervalId = setInterval(fetchLogs, 1000);
        const clearIntervalId = setInterval(clearServerLogs, 5000);

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
                        (() => {
                            // Group logs by timestamp, combining object + position
                            const grouped = new Map();
                            logs.forEach(log => {
                                if (log.type === 'detection' && /\b(left|right|middle)\b/i.test(log.message)) {
                                    const posMatch = log.message.match(/\b(left|right|middle)\b/i);
                                    const typeMatch = log.message.match(/Detected (\w+)/);
                                    const pos = posMatch ? posMatch[1].toLowerCase() : null;
                                    const objType = typeMatch ? typeMatch[1] : 'object';
                                    if (pos) {
                                        if (!grouped.has(log.timestamp)) {
                                            grouped.set(log.timestamp, { detections: [], type: log.type });
                                        }
                                        // Store as "object position" string, dedupe
                                        const detection = `${objType} ${pos}`;
                                        if (!grouped.get(log.timestamp).detections.includes(detection)) {
                                            grouped.get(log.timestamp).detections.push(detection);
                                        }
                                    }
                                } else if (log.type !== 'detection') {
                                    // Non-detection logs keep their own entry
                                    grouped.set(log.timestamp + log.message, { message: log.message, type: log.type, timestamp: log.timestamp });
                                }
                            });

                            // Convert to array and sort by timestamp descending
                            return Array.from(grouped.entries())
                                .map(([key, value]) => ({
                                    timestamp: value.timestamp || key,
                                    type: value.type,
                                    detections: value.detections || null,
                                    message: value.message
                                }))
                                .sort((a, b) => b.timestamp.localeCompare(a.timestamp))
                                .map((entry, index) => (
                                    <div key={index} className={`log-entry log-${entry.type}`}>
                                        <span className="log-timestamp">{formatTime(entry.timestamp)}</span>
                                        <span className="log-message">
                                            {entry.detections
                                                ? `Detected: [${entry.detections.join(', ')}]`
                                                : entry.message
                                            }
                                        </span>
                                    </div>
                                ));
                        })()
                    )}
                </div>
            </div>
        </div>
    )
}

export default DepthView
