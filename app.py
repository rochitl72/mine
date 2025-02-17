from flask import Flask, render_template, Response, request, jsonify
import cv2
import sqlite3
import os
import threading

app = Flask(__name__)

# OpenCV Video Capture
cap = cv2.VideoCapture(0)  # First camera
cap2 = None  # Placeholder for second camera (if needed)

def generate_frames(camera):
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed_1')
def video_feed_1():
    return Response(generate_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_2')
def video_feed_2():
    if cap2:
        return Response(generate_frames(cap2), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "No second camera available"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_detected_persons', methods=['GET'])
def get_detected_persons():
    conn = sqlite3.connect('tom_base.db')
    cursor = conn.cursor()
    cursor.execute("SELECT track_id, name, clothing_color, last_location, image_path, timestamp FROM tracked_people")
    people = cursor.fetchall()
    conn.close()
    detected = []
    for p in people:
        detected.append({
            'id': p[0],
            'name': p[1],
            'clothing_color': p[2],
            'last_location': p[3],
            'image_path': p[4],
            'timestamp': p[5]
        })
    return jsonify(detected)

@app.route('/delete_person', methods=['POST'])
def delete_person():
    data = request.json
    track_id = data.get('track_id')
    conn = sqlite3.connect('tom_base.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tracked_people WHERE track_id = ?", (track_id,))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
