import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import numpy as np
from sort import Sort
import torch
import time
from threading import Thread
import multiprocessing
from threading import Lock

class TrackedPerson:
    def __init__(self, tracking_id):
        self.tracking_id = tracking_id
        self.prev_position = None
        self.entered = set()
        self.exited = set()

def is_inside_roi(x, y, roi):
    return roi[1] < y < roi[3]

def track(tracker, frame, yolo_detections, roi, total_count, entered_count, exited_count, tracked_people, lock):
    # Convert YOLOv5 detections to SORT format
    sort_detections = []
    if not yolo_detections.empty:
        for index, row in yolo_detections.iterrows():
            if int(row[5]) == 0 and row[4] > 0.5:  # İnsanları sınıflandır
                x1, y1, x2, y2, confidence = int(row[0]), int(row[1]), int(row[2]), int(row[3]), row[4]
                sort_detections.append([x1, y1, x2, y2, confidence])

    # Convert the list to a NumPy array
    if not sort_detections:
        sort_detections = np.empty((0, 5))
    else:
        sort_detections = np.array(sort_detections)

    # Update the SORT tracker
    tracking_results = tracker.update(sort_detections)

    # Draw bounding boxes for tracked objects
    for track in tracking_results:
        tracking_id = int(track[4])
        x1, y1, x2, y2 = track[:4]

        # Check if the object is inside the ROI
        if is_inside_roi((x1 + x2) / 2, (y1 + y2) / 2, roi):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, str(tracking_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check the direction of movement
            person = tracked_people.get(tracking_id, TrackedPerson(tracking_id))
            if person.prev_position is not None:
                if y2 > person.prev_position[3]:  # Object moving from top to bottom (exiting)
                    with lock:
                        if tracking_id not in person.exited:
                            person.exited.add(tracking_id)
                            exited_count.value += 1
                elif y1 < person.prev_position[1]:  # Object moving from bottom to top (entering)
                    with lock:
                        if tracking_id not in person.entered:
                            person.entered.add(tracking_id)
                            entered_count.value += 1

            person.prev_position = (x1, y1, x2, y2)
            tracked_people[tracking_id] = person

    total_count.value = entered_count.value - exited_count.value

    # Display total counts in the top left corner of the frame
    cv2.putText(frame, f"Entered: {entered_count.value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Exited: {exited_count.value}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Total: {total_count.value}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

def detection(source, window_name, roi, total_count, entered_count, exited_count, tracked_people, lock):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on " + window_name, device)

    # Load the YOLO model
    model = YOLO('yolov8s.pt').to(device)

    cap = cv2.VideoCapture("busfinal.mp4")

    tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.30)

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Define the ROI
        roi_coordinates = (90, 200, 500, 230)  # Update with your desired ROI coordinates
        frame = cv2.rectangle(frame, (roi_coordinates[0], roi_coordinates[1]), (roi_coordinates[2], roi_coordinates[3]),
                              (255, 0, 0), 2)

        results = model.predict(frame)
        a = results[0].boxes.data
        yolo_detections = pd.DataFrame(a.cpu().numpy()).astype("float")

        frame = track(tracker, frame, yolo_detections, roi_coordinates, total_count, entered_count, exited_count, tracked_people, lock)

        frame = cv2.resize(frame, (640, 480))
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        print(window_name + " FPS = " + str(1 / (time.time() - t0)))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    window_name1 = "first_cam"
    window_name2 = "second_cam"

    roi_coordinates = (90, 200, 500, 230)  # Update with your desired ROI coordinates
    total_count = multiprocessing.Value('i', 0)
    entered_count = multiprocessing.Value('i', 0)
    exited_count = multiprocessing.Value('i', 0)
    tracked_people = {}  # Changed to use a dictionary to store TrackedPerson objects
    lock = Lock()

    t1 = Thread(target=detection, args=(0, window_name1, roi_coordinates, total_count, entered_count, exited_count, tracked_people, lock))
    t1.start()
    t1.join()
