from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
from collections import deque

app = Flask(__name__)
CORS(app)

# Store logs (keep last 100 entries)
detection_logs = deque(maxlen=100)

def add_log(message, log_type="info"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detection_logs.append({
        "timestamp": timestamp,
        "type": log_type,
        "message": message
    })

@app.route('/logs', methods=['GET'])
def get_logs():
    return jsonify(list(detection_logs))

@app.route('/logs', methods=['POST'])
def post_log():
    data = request.get_json()
    message = data.get('message', '')
    log_type = data.get('type', 'info')
    add_log(message, log_type)
    return jsonify({"status": "ok"})

@app.route('/logs/clear', methods=['GET', 'POST'])
def clear_logs():
    detection_logs.clear()
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    print("Starting log server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, threaded=True)
