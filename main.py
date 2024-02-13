import os
import random
import cv2
from ultralytics import YOLO
from tracker import Tracker

video_path = "data/f1.mp4"
video_out_path = os.path.join('data', 'f1-out.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for Mac

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap_out = cv2.VideoWriter(video_out_path, fourcc, fps, (frame_width, frame_height))

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
    
model = YOLO("yolov8n.pt")
detection_threshold = 0.5
tracker = Tracker()

while ret:

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            print(r)
            *cords, score, class_id = r
            cords = [int(x) for x in cords]

            if score > detection_threshold:
                detections.append([*cords, score])
            
            
        tracker.update(frame, detections)
        
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)


    #cap_out.write(frame)
    # Display the frame using Matplotlib
    cap_out.write(frame)
    ret, frame = cap.read()

# Release the video capture object
cap.release()
cap_out.release()
cv2.destroyAllWindows()