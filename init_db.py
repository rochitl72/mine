import sqlite3
import datetime

# Database file
DB_FILE = "tom_base.db"

def create_database():
    """Creates the database and table if they do not exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tracked_people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER UNIQUE,
            name TEXT DEFAULT 'Unknown',
            face_embedding BLOB,
            clothing_color TEXT,
            last_location TEXT,
            image_path TEXT,
            timestamp TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database and table created successfully.")

def insert_tracked_person(face_embedding, clothing_color, last_location, image_path):
    """Inserts a new tracked person into the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute('''
        INSERT INTO tracked_people (face_embedding, clothing_color, last_location, image_path, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (face_embedding, clothing_color, last_location, image_path, timestamp))

    conn.commit()
    conn.close()
    print("Tracked person added successfully.")

# Run the script to create the database
if __name__ == "__main__":
    create_database()

    # Example: Insert a test entry
    fake_embedding = b'123456'  # Example binary data for face embedding
    insert_tracked_person(fake_embedding, "Blue", "Cam 1", "faces/person_1.jpg")
