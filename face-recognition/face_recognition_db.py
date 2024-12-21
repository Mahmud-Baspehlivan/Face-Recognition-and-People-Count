import face_recognition
import cv2
import os
import glob
import numpy as np
import sqlite3
from datetime import datetime

class FaceRecDB:
    def __init__(self, db_path="face_recognition.db"):
        self.db_path = db_path
        self.frame_resizing = 0.25
        self.setup_database()
        self.load_encodings_from_db()
        self.alert_cooldown = 60  # Saniye cinsinden uyarı bekleme süresi
        self.last_alerts = {}  # Son uyarıların zamanını takip etmek için

    def migrate_database(self):
        """Veritabanını yeni yapıya günceller"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Mevcut tabloları yedekle
        try:
            c.execute("ALTER TABLE persons RENAME TO persons_old")
        except sqlite3.OperationalError:
            pass  # Tablo zaten yok
            
        # Yeni tabloları oluştur
        c.execute('''CREATE TABLE IF NOT EXISTS persons
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     name TEXT NOT NULL,
                     security_level INTEGER DEFAULT 1,
                     access_zones TEXT DEFAULT 'public',
                     time_restrictions TEXT,
                     is_blacklisted INTEGER DEFAULT 0,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
                     
        # Eski veriler varsa, yeni tabloya aktar
        try:
            c.execute("INSERT INTO persons (id, name, created_at) SELECT id, name, created_at FROM persons_old")
            c.execute("DROP TABLE persons_old")
        except sqlite3.OperationalError:
            pass  # Eski tablo yok
            
        conn.commit()
        conn.close()

    def setup_database(self):
        """Veritabanı ve gerekli tabloları oluşturur"""
        # Önce veritabanını migrate et
        self.migrate_database()
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Zaman kısıtlamaları için format: "09:00-17:00,Mon-Fri"
        
        # Alarm/Uyarı log tablosu
        c.execute('''CREATE TABLE IF NOT EXISTS security_alerts
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     alert_type TEXT NOT NULL,
                     person_id INTEGER,
                     description TEXT,
                     severity INTEGER,
                     handled INTEGER DEFAULT 0,
                     timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     FOREIGN KEY (person_id) REFERENCES persons (id))''')
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Kişiler tablosu
        c.execute('''CREATE TABLE IF NOT EXISTS persons
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     name TEXT NOT NULL,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        # Yüz kodları tablosu
        c.execute('''CREATE TABLE IF NOT EXISTS face_encodings
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     person_id INTEGER,
                     encoding BLOB NOT NULL,
                     FOREIGN KEY (person_id) REFERENCES persons (id))''')
        
        # Tanıma geçmişi tablosu
        c.execute('''CREATE TABLE IF NOT EXISTS recognition_history
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     person_id INTEGER,
                     confidence REAL,
                     timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     FOREIGN KEY (person_id) REFERENCES persons (id))''')
        
        conn.commit()
        conn.close()

    def add_person(self, name, image_path):
        """Yeni bir kişi ve yüz kodlamasını veritabanına ekler"""
        # Resmi yükle ve kodla
        img = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(rgb_img)[0]
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            # Kişiyi ekle
            c.execute("INSERT INTO persons (name) VALUES (?)", (name,))
            person_id = c.lastrowid
            
            # Yüz kodlamasını ekle
            encoding_bytes = face_encoding.tobytes()
            c.execute("INSERT INTO face_encodings (person_id, encoding) VALUES (?, ?)",
                     (person_id, encoding_bytes))
            
            conn.commit()
            print(f"Kişi başarıyla eklendi: {name}")
        except Exception as e:
            print(f"Hata oluştu: {e}")
            conn.rollback()
        finally:
            conn.close()
        
        # Kodlamaları yeniden yükle
        self.load_encodings_from_db()

    def load_encodings_from_db(self):
        """Veritabanından tüm yüz kodlamalarını yükler"""
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""SELECT p.id, p.name, fe.encoding 
                    FROM persons p 
                    JOIN face_encodings fe ON p.id = fe.person_id""")
        
        for person_id, name, encoding_bytes in c.fetchall():
            face_encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            self.known_face_ids.append(person_id)
        
        conn.close()
        print(f"{len(self.known_face_names)} kişi yüklendi.")

    def detect_known_faces(self, frame):
        """Karedeki yüzleri tespit eder ve tanımlar"""
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            person_id = None

            if True in matches:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    person_id = self.known_face_ids[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    
                    # Tanıma geçmişini kaydet
                    self.log_recognition(person_id, confidence)

            face_names.append(name)

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

    def check_access_permission(self, person_id):
        """Kişinin erişim izinlerini kontrol eder"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""SELECT security_level, access_zones, time_restrictions, is_blacklisted
                    FROM persons WHERE id = ?""", (person_id,))
        result = c.fetchone()
        conn.close()
        
        if not result:
            return False, "Kişi bulunamadı"
            
        security_level, access_zones, time_restrictions, is_blacklisted = result
        
        if is_blacklisted:
            return False, "Kara listede"
            
        # Zaman kısıtlamalarını kontrol et
        if time_restrictions:
            current_time = datetime.now().time()
            current_day = datetime.now().strftime('%a')
            
            # Format: "09:00-17:00,Mon-Fri"
            time_range, days = time_restrictions.split(',')
            start_time, end_time = time_range.split('-')
            allowed_days = days.split('-')
            
            start = datetime.strptime(start_time, '%H:%M').time()
            end = datetime.strptime(end_time, '%H:%M').time()
            
            if current_day not in allowed_days or not (start <= current_time <= end):
                return False, "Zaman kısıtlaması"
        
        return True, "Erişim izni var"

    def log_security_alert(self, alert_type, person_id, description, severity):
        """Güvenlik uyarısını kaydeder"""
        current_time = datetime.now()
        
        # Uyarı spam'ini önle
        alert_key = f"{alert_type}_{person_id}"
        if alert_key in self.last_alerts:
            if (current_time - self.last_alerts[alert_key]).seconds < self.alert_cooldown:
                return
        
        self.last_alerts[alert_key] = current_time
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""INSERT INTO security_alerts 
                    (alert_type, person_id, description, severity)
                    VALUES (?, ?, ?, ?)""",
                 (alert_type, person_id, description, severity))
        
        conn.commit()
        conn.close()

    def log_recognition(self, person_id, confidence):
        """Tanıma olayını veritabanına kaydeder ve güvenlik kontrollerini yapar"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""INSERT INTO recognition_history (person_id, confidence)
                    VALUES (?, ?)""", (person_id, float(confidence)))
        
        conn.commit()
        conn.close()

    def remove_duplicate_persons(self):
        """Aynı isme sahip kayıtları kontrol eder ve yinelenen kayıtları siler"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Aynı isme sahip kayıtları bul
        c.execute("""SELECT name, COUNT(*) as count, GROUP_CONCAT(id) as ids
                    FROM persons 
                    GROUP BY name 
                    HAVING count > 1""")
        
        duplicates = c.fetchall()
        
        for name, count, ids in duplicates:
            id_list = ids.split(',')
            # İlk kaydı tut, diğerlerini sil
            keep_id = id_list[0]
            delete_ids = id_list[1:]
            
            # İlgili yüz kodlamalarını ve geçmiş kayıtları sil
            for delete_id in delete_ids:
                c.execute("DELETE FROM face_encodings WHERE person_id = ?", (delete_id,))
                c.execute("DELETE FROM recognition_history WHERE person_id = ?", (delete_id,))
                c.execute("DELETE FROM persons WHERE id = ?", (delete_id,))
            
            print(f"'{name}' için {len(delete_ids)} yinelenen kayıt silindi.")
        
        conn.commit()
        conn.close()
        
        # Kodlamaları yeniden yükle
        self.load_encodings_from_db()

    def get_recognition_history(self, days=7):
        """Son x günün tanıma geçmişini getirir"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""SELECT p.name, rh.confidence, rh.timestamp
                    FROM recognition_history rh
                    JOIN persons p ON rh.person_id = p.id
                    WHERE rh.timestamp >= datetime('now', ?)
                    ORDER BY rh.timestamp DESC""", (f'-{days} days',))
        
        history = c.fetchall()
        conn.close()
        return history