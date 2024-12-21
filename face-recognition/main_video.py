import cv2
from face_recognition_db import FaceRecDB
import time
from datetime import datetime

def main():
    # FaceRecDB sınıfını başlat
    face_rec = FaceRecDB()
    
    # face_rec.add_person("Mahmud", "images/mahmud.jpg")
    # face_rec.add_person("Osman", "images/osman.jpg")
    # face_rec.add_person("Elon Musk", "images/Elon Musk.jpg")
    # face_rec.add_person("Jeff Bezoz", "images/Jeff Bezoz.jpg")
    # face_rec.add_person("Messi", "images/Messi.webp")
    # face_rec.add_person("Ryan Reynolds", "images/Ryan Reynolds.jpg")
    
    # Yinelenen kayıtları temizle
    face_rec.remove_duplicate_persons()

    # Kamera başlat
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows için DirectShow kullan
    
    # Kamera açılamadıysa hata ver
    if not cap.isOpened():
        print("Hata: Kamera açılamadı!")
        return
    
    # FPS hesaplama için değişkenler
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # FPS hesapla
        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Yüzleri tespit et
        face_locations, face_names = face_rec.detect_known_faces(frame)
        
        # Tespit edilen yüzleri çerçevele ve isimleri yaz
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            
            # Erişim kontrolü (tanınan kişiler için)
            if name != "Unknown":
                person_id = face_rec.known_face_ids[face_rec.known_face_names.index(name)]
                has_access, reason = face_rec.check_access_permission(person_id)
                
                # Renk ve durum mesajı belirle
                if has_access:
                    color = (0, 255, 0)  # Yeşil
                    status = "Authorized"
                else:
                    color = (0, 0, 255)  # Kırmızı
                    status = f"Denied: {reason}"
                    # Güvenlik uyarısı oluştur
                    face_rec.log_security_alert(
                        "unauthorized_access",
                        person_id,
                        f"Yetkisiz erişim girişimi: {reason}",
                        severity=2
                    )
            else:
                color = (0, 165, 255)  # Turuncu
                status = "Unknown"
                # Bilinmeyen kişi uyarısı
                face_rec.log_security_alert(
                    "unknown_person",
                    None,
                    "Tanınmayan kişi tespit edildi",
                    severity=1
                )

            # Kişi bilgilerini ekrana yaz
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, f"{name} - {status}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

        # Ekranın üst kısmına bilgi paneli ekle
        info_text = f"Time: {current_time} | FPS: {fps}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Security System", frame)

        # ESC tuşuna basılınca çık
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()