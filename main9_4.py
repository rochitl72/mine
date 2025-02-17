import cv2
import torch
import numpy as np
import sqlite3
import os
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Determine the best available device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"Using device: {device}")

# Initialize YOLO Model and DeepSORT Tracker
model = YOLO("yolov8n.pt").to(device)
tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.5)

# Initialize Face Recognition Model
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Ensure Faces Directory Exists
os.makedirs("faces", exist_ok=True)

# Connect to SQLite Database
db_path = "tom_base.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

# Create Table with Indexing
cursor.execute("""
CREATE TABLE IF NOT EXISTS tracked_people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER UNIQUE,
    name TEXT DEFAULT 'Unknown',
    face_embedding BLOB,
    clothing_color TEXT,
    last_location TEXT,
    image_path TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
)
""")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_face_embedding ON tracked_people(face_embedding);")
conn.commit()

# Open Two Cameras
cap1 = cv2.VideoCapture(0)  # First camera (Camera 1)
cap2 = cv2.VideoCapture(1)  # Second camera (Camera 2)

# Set camera properties
for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both webcams.")
    exit()

# Function to Detect Clothing Color and Map to Name
def detect_clothing_color(frame, bbox):
    x1, y1, x2, y2 = bbox
    clothing_region = frame[y2 - int((y2 - y1) * 0.3):y2, x1:x2]
    
    if clothing_region.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(clothing_region, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    dominant_hue = int(np.argmax(hist))

    # Mapping HSV Hue to Color Names
    def hue_to_color(hue):
        if 0 <= hue < 15 or hue >= 165:
            return "Red"
        elif 15 <= hue < 35:
            return "Orange"
        elif 35 <= hue < 85:
            return "Yellow/Green"
        elif 85 <= hue < 125:
            return "Cyan/Blue"
        elif 125 <= hue < 165:
            return "Purple"
        else:
            return "Unknown"

    return hue_to_color(dominant_hue)

# Function to Match Faces Using Cosine Similarity
def match_face(face_embedding, threshold=0.7):
    cursor.execute("SELECT track_id, face_embedding FROM tracked_people WHERE face_embedding IS NOT NULL")
    records = cursor.fetchall()
    
    for track_id, db_embedding in records:
        db_embedding = np.frombuffer(db_embedding, dtype=np.float32)
        if cosine_similarity([face_embedding], [db_embedding])[0][0] >= threshold:
            return track_id  # Return existing track ID if face matches
    
    return None  # No match found

# Function to Process Frame
def process_frame(frame, camera_id):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame, conf=0.5, iou=0.5, classes=[0])
    
    detections = [([int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])], 
                   box.conf[0].item(), "person")
                  for r in results for box in r.boxes]
    
    tracks = tracker.update_tracks(detections, frame=frame)
    faces = app.get(rgb_frame)
    track_ids_in_frame = set()
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        track_ids_in_frame.add(track_id)
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        clothing_color = detect_clothing_color(frame, (x1, y1, x2, y2))
        image_path, face_embedding, name = "", None, "Unknown"

        # Face Recognition & ID Assignment
        for face in faces:
            fx1, fy1, fx2, fy2 = map(int, face.bbox)
            if x1 < fx1 < x2 and y1 < fy1 < y2:
                face_embedding = face.normed_embedding
                matched_id = match_face(face_embedding)
                if matched_id:
                    track_id = matched_id  # Assign existing track ID if face matches
                    name = f"Person_{track_id}"
                break

        # Update or Insert Person into Database
        cursor.execute("""
            INSERT INTO tracked_people (track_id, name, face_embedding, clothing_color, last_location, image_path, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(track_id) DO UPDATE SET
                name=excluded.name,
                face_embedding=excluded.face_embedding,
                clothing_color=excluded.clothing_color,
                last_location=?,
                image_path=excluded.image_path,
                timestamp=?
        """, (track_id, name, 
              face_embedding.tobytes() if face_embedding is not None else None, 
              clothing_color, camera_id, image_path, 
              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              camera_id,  # Update last_location explicitly
              datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
    
    # Remove people from database if they are not in the current frame
    cursor.execute("DELETE FROM tracked_people WHERE track_id NOT IN ({})".format(
        ','.join(map(str, track_ids_in_frame)) if track_ids_in_frame else "NULL"))
    conn.commit()

# Function to Handle Two Cameras Simultaneously
def camera_loop():
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 and not ret2:
            print("Error: Could not read frames from both cameras.")
            break

        if ret1:
            process_frame(frame1, "Camera_1")
        
        if ret2:
            process_frame(frame2, "Camera_2")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start Camera Loop
if __name__ == "__main__":
    camera_loop()
    cap1.release()
    cap2.release()
    conn.close()
    cv2.destroyAllWindows()
