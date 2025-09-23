from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np

class VehicleCounter:
    def __init__(self, name, x1, x2, y):
        self.name = name
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.vehicle_count = []

    def draw_line(self, img, color=(0, 0, 255)):
        cv2.line(img, (self.x1, self.y), (self.x2, self.y), color, 3)

    def check_and_count(self, cx, cy, id, img):
        if self.x1 < cx < self.x2 and self.y - 10 < cy < self.y + 10:
            if self.vehicle_count.count(id) == 0:
                self.vehicle_count.append(id)
                # Flash the specific segment green
                self.draw_line(img, color=(0, 255, 0))
                return True
        return False

model = YOLO('./Vehicle_detection/yolo_weights/yolov8n.pt')

cap = cv2.VideoCapture('./videos/video2.mp4')

mask1 = cv2.imread('./images/mask1.jpg')
mask2 = cv2.imread('./images/mask2.jpg')

# Tracking vehicles (single tracker for simplicity, since regions are separate and IDs won't conflict)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Create counters for left (incoming) and right (outgoing)
incoming_counter = VehicleCounter("Incoming", 150, 850, 450)  # Left side
outgoing_counter = VehicleCounter("Outgoing", 900, 1700, 450)  # Right side

while True:
    isFrame, img = cap.read()
    if not isFrame:
        break
    imgRegion1 = cv2.bitwise_and(img, mask1)  # For left side
    imgRegion2 = cv2.bitwise_and(img, mask2)  # For right side
    results1 = model(imgRegion1, stream=True)
    results2 = model(imgRegion2, stream=True)

    detections = np.empty((0, 5))

    # Process detections from both regions (combine them for unified tracking)
    for results_set in [results1, results2]:
        for r in results_set:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Mark the confidence of the object detection
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Get the class names from the YOLO model
                cls_id = int(box.cls[0])  # class index
                current_class = model.names[cls_id]
                vehicles_list = ["car", "bus", "motorcycle", "truck"]

                if current_class in vehicles_list and conf >= 0.3:
                    # Confidence text (optional, as in your original code)
                    cvzone.putTextRect(img, f'{conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2, offset=10)

                    # Current id'd of the vehicle
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    # Append those id's to numpy array
                    detections = np.vstack((detections, currentArray))

    # Maintaining the track
    resultsTracker = tracker.update(detections)

    # Draw the counting lines (segments in red)
    incoming_counter.draw_line(img)
    outgoing_counter.draw_line(img)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), t=4, l=20, rt=3)

        # Find the center of the vehicles or bounding box
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (0, 0, 225), cv2.FILLED)

        # Check for crossing on each side
        incoming_counter.check_and_count(cx, cy, id, img)
        outgoing_counter.check_and_count(cx, cy, id, img)

    # Display counts
    cvzone.putTextRect(img, f'{incoming_counter.name}: {len(incoming_counter.vehicle_count)}', (50, 50))
    cvzone.putTextRect(img, f'{outgoing_counter.name}: {len(outgoing_counter.vehicle_count)}', (1550, 50))

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()