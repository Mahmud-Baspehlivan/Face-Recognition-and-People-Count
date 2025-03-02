import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import numpy as np
from sort import Sort
import torch
import time
from threading import Thread, Lock
import multiprocessing
from face_recognition_db import FaceRecDB
from datetime import datetime

class SecuritySystem:
    def __init__(self):
        # Yüz tanıma için gerekli değişkenler
        self.face_rec = FaceRecDB()
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.fps = 0
        
        # İnsan sayma için gerekli değişkenler
        self.total_count = multiprocessing.Value('i', 0)
        self.entered_count = multiprocessing.Value('i', 0)
        self.exited_count = multiprocessing.Value('i', 0)
        self.tracked_people = {}
        self.lock = Lock()
        self.roi_coordinates = (100, 730, 650, 760)
        
        # YOLO ve SORT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO('yolov8s.pt').to(self.device)
        self.tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.30)

    def face_recognition_process(self):
        """Yüz tanıma sistemi - Birinci kamera için"""
        cap = cv2.VideoCapture(0)  # Birinci kamera

        # self.face_rec.add_person("Mahmud", "images/mahmud.jpg")
        # self.face_rec.add_person("Osman", "images/osman.jpg")
        # self.face_rec.add_person("Elon Musk", "images/Elon Musk.jpg")
        # self.face_rec.add_person("Jeff Bezoz", "images/Jeff Bezoz.jpg")
        # self.face_rec.add_person("Messi", "images/Messi.webp")
        # self.face_rec.add_person("Ryan Reynolds", "images/Ryan Reynolds.jpg")
        # self.face_rec.add_person("Cemil Öz", "images/cemil öz.jpg")
        
        if not cap.isOpened():
            print("Hata: Face Recognition kamerası açılamadı!")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FPS hesaplama
            self.fps_counter += 1
            if (time.time() - self.fps_start_time) > 1:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Yüz tespiti ve tanıma
            face_locations, face_names = self.face_rec.detect_known_faces(frame)
            
            # Tespit edilen yüzleri işaretle
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                
                if name != "Unknown":
                    person_id = self.face_rec.known_face_ids[self.face_rec.known_face_names.index(name)]
                    has_access, reason = self.face_rec.check_access_permission(person_id)
                    color = (0, 255, 0) if has_access else (0, 0, 255)
                    status = "Authorized" if has_access else f"Denied: {reason}"
                    
                    if not has_access:
                        self.face_rec.log_security_alert(
                            "unauthorized_access",
                            person_id,
                            f"Yetkisiz erişim girişimi: {reason}",
                            severity=2
                        )
                else:
                    color = (0, 165, 255)
                    status = "Unknown"
                    self.face_rec.log_security_alert(
                        "unknown_person",
                        None,
                        "Tanınmayan kişi tespit edildi",
                        severity=1
                    )

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                cv2.putText(frame, f"{name} - {status}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

            cv2.putText(frame, f"Time: {current_time} | FPS: {self.fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Face Recognition System", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()

    def people_counting_process(self):
        """İnsan sayma sistemi - İkinci kamera için"""
        cap = cv2.VideoCapture("guvenlik.mp4")  # İkinci kamera
        if not cap.isOpened():
            print("Hata: People Counting kamerası açılamadı!")
            return

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # ROI çizimi
            frame = cv2.rectangle(frame, 
                                (self.roi_coordinates[0], self.roi_coordinates[1]), 
                                (self.roi_coordinates[2], self.roi_coordinates[3]),
                                (255, 0, 0), 2)

            # YOLO tespitleri
            results = self.model.predict(frame)
            detections = pd.DataFrame(results[0].boxes.data.cpu().numpy()).astype("float")

            # SORT için detections hazırla
            sort_detections = []
            if not detections.empty:
                for _, row in detections.iterrows():
                    if int(row[5]) == 0 and row[4] > 0.5:
                        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                        confidence = row[4]
                        sort_detections.append([x1, y1, x2, y2, confidence])

            # SORT tracker güncelle
            if not sort_detections:
                sort_detections = np.empty((0, 5))
            else:
                sort_detections = np.array(sort_detections)

            tracking_results = self.tracker.update(sort_detections)

            # İzlenen nesneleri işle
            for track in tracking_results:
                tracking_id = int(track[4])
                x1, y1, x2, y2 = track[:4]
                
                if self.roi_coordinates[1] < (y1 + y2)/2 < self.roi_coordinates[3]:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, str(tracking_id), (int(x1), int(y1) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Sayaç güncelleme
                    with self.lock:
                        if tracking_id not in self.tracked_people:
                            self.tracked_people[tracking_id] = {"last_y": (y1 + y2)/2}
                        else:
                            last_y = self.tracked_people[tracking_id]["last_y"]
                            if (y1 + y2)/2 > last_y and "counted_down" not in self.tracked_people[tracking_id]:
                                self.exited_count.value += 1
                                self.tracked_people[tracking_id]["counted_down"] = True
                            elif (y1 + y2)/2 < last_y and "counted_up" not in self.tracked_people[tracking_id]:
                                self.entered_count.value += 1
                                self.tracked_people[tracking_id]["counted_up"] = True
                            self.tracked_people[tracking_id]["last_y"] = (y1 + y2)/2

            # Sayaçları göster
            cv2.putText(frame, f"Giren: {self.entered_count.value}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Cikan: {self.exited_count.value}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Mevcut Sayi: {self.entered_count.value - self.exited_count.value}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("People Counting System", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            print(f"People Counting FPS: {1/(time.time() - t0):.1f}")

        cap.release()

    def run(self):
        # İki sistemi ayrı thread'lerde başlat
        t1 = Thread(target=self.face_recognition_process)
        t2 = Thread(target=self.people_counting_process)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = SecuritySystem()
    system.run()
