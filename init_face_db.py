import sqlite3

def create_database():
    conn = sqlite3.connect("face.db")
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB
        )
    ''')
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
    print("Database 'face.db' initialized successfully.")