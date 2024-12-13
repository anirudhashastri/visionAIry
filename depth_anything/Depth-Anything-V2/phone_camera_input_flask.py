'''
Author: Elio Khouri, Josef LaFranchise, Anirudha Shastri, Karthik Koduru
Date: 11/22/2024
CS 7180: Advanced Perception
phone_camera_input_flask.py
'''
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import queue
import threading
import time
import socket

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False)

# Thread-safe queue to hold incoming frames
frame_queue = queue.Queue(maxsize=10)

# Throttle variables
last_frame_time = 0  # Last processed frame timestamp
throttle_interval = 1 / 3  # Process at 3 FPS (333ms)

@app.route('/')
def index():
    """
    The index page.
    """
    return "Flask server is running!", 200

@socketio.on('connect')
def on_connect():
    """
    Sends a response when client connects.
    """
    print("WebSocket: Client connected.")
    emit('ack', {'message': 'Connected to server!'})

@socketio.on('disconnect')
def on_disconnect():
    """
    Triggered when a client disconnects.
    """
    print("WebSocket: Client disconnected.")

@socketio.on('frame')
def handle_frame(data):
    """
    Takes a incoming frame and places it in the frame queue. Also throttles the
    amount of frames that are added to the queue.
    
    Args:
        data - The frame data
    """
    global last_frame_time

    # Get current time
    current_time = time.time()

    # Enforce throttling interval
    if current_time - last_frame_time < throttle_interval:
        if current_time - last_frame_time > 1:  # Log only once every second
            print("Frame dropped due to throttling.")
        return  # Drop the frame if within throttle interval

    last_frame_time = current_time  # Update the last processed frame time

    # Decode and process the frame
    try:
        if current_time - last_frame_time > 1:  # Log only once every second
            print(f"WebSocket: Received frame of size {len(data)} bytes. Processing...")
        np_img = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Add to the queue if space is available
        if not frame_queue.full():
            frame_queue.put(frame)
        emit('ack', {'message': 'Frame received'})
    except Exception as e:
        print(f"Error processing frame: {e}")

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """
    Takes a incoming frame from a POST request and places it in the frame queue. 
    Also throttles the amount of frames that are added to the queue.
    
    """
    global last_frame_time

    # Get current time
    current_time = time.time()

    # Enforce throttling interval
    if current_time - last_frame_time < throttle_interval:
        if current_time - last_frame_time > 1:  # Log only once every second
            print("Frame dropped due to throttling.")
        return jsonify({'message': 'Frame dropped'}), 429  # HTTP 429 Too Many Requests

    last_frame_time = current_time  # Update the last processed frame time

    # Handle frame data
    try:
        file = request.files.get('frame')
        if not file:
            return jsonify({'error': 'No frame provided'}), 400

        np_img = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Add to the queue if space is available
        if not frame_queue.full():
            frame_queue.put(frame)
        return jsonify({'message': 'Frame received'}), 200
    except Exception as e:
        if current_time - last_frame_time > 1:  # Log only once every second
            print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

def get_latest_frame():
    """
    Returns the next frame in the frame queue.
    
    Returns:
        frame - The next frame, None if no frames in frame queue
    """
    global last_frame_time  # To track the time since last log
    if frame_queue.empty():
        # Only log every 1 second to reduce spam
        current_time = time.time()
        if current_time - last_frame_time > 1:  # Log every 1 second
            print("No frames in the queue.")
            last_frame_time = current_time
        time.sleep(0.1)  # Add a small delay to avoid tight looping
        return None
    # Get the frame from the queue
    frame = frame_queue.get()

    # Step 1: Normalize raw frame orientation
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # Rotate raw frame to correct orientation

    return frame

def start_server():
    """
    Initializes the server.
    """
    # Get the local IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    port = 5001  # Default port

    # Print server details
    print(f"[+] Flask server is starting...")
    print(f"[+] Server running on:")
    print(f"    - Local:   http://127.0.0.1:{port}")
    print(f"    - Network: http://{local_ip}:{port}")
    print(f"[+] WebSocket available at ws://{local_ip}:{port}")

    socketio.run(app, host='0.0.0.0', port=5001, debug=False, use_reloader=False)

def broadcast_ip(port=5001):
    """
    Broadcast IP address and port. Broadcast at an interval of every 2 seconds.
    
    Args:
        port - Port to use
    """
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    local_ip = socket.gethostbyname(socket.gethostname())
    message = f"{local_ip}:{port}"  # Broadcast IP and port

    print(f"[+] Broadcasting server IP: {local_ip}:{port}")

    while True:
        udp_socket.sendto(message.encode(), ('<broadcast>', 37020))
        time.sleep(2)  # Broadcast every 2 seconds

if __name__ == '__main__':
    """
    Main entry point of the program. Initializes server and broadcasts IP.
    """
    threading.Thread(target=broadcast_ip, daemon=True).start()
    start_server()
