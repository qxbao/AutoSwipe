import sqlite3
import os

DB_PATH = os.path.dirname(__file__) + '/../data/swipes.db'

class Database:
    db = None
    def __init__(self):
        self.db = sqlite3.connect()
        return self

    def close(self):
        if self.db:
            self.db.close()
            self.db = None

    def save_profile(self, profile_id: str, age: int, num_images: int, profile_folder: str, preference_score: float) -> None:
        cursor = self.db.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO profiles (profile_id, age, num_images, profile_folder, preference_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (profile_id, age, num_images, profile_folder, preference_score))
        self.db.commit()
    
    @staticmethod
    def save_profile_folder(profile_id, images) -> str:
        profile_folder = os.path.join(os.path.dirname(__file__), '../data/images', profile_id)
        if not os.path.exists(profile_folder):
            os.makedirs(profile_folder)
        for i, image in enumerate(images):
            image_path = os.path.join(profile_folder, f'image_{i}.jpg')
            with open(image_path, 'wb') as img_file:
                img_file.write(image)
        return profile_folder
    
    @staticmethod
    def init_db() -> None:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT UNIQUE,
                age INTEGER,
                num_images INTEGER,
                profile_folder TEXT,
                preference_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profile_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT,
                image_path TEXT,
                image_order INTEGER,
                FOREIGN KEY (profile_id) REFERENCES profiles (profile_id)
            )
        ''')
        conn.commit()
        conn.close()
    