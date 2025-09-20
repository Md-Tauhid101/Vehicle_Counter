from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np

model = YOLO('./Vehicle_detection/yolo_weights/yolov8n.pt')

cap = cv2.VideoCapture('./videos/video.mp4')

mask = cv2.imread('./images/mask1.jpg')

# Tracking vehicles
tracker = Sort(max_age=20, min_hits=3, iou_threshold= 0.3)

# limits = [850, 500, 200, 490]
line_y = 450  # adjust after testing
line_x1, line_x2 = 150, 850   # spanning the road width
vehicle_count = []

while True:
    isFrame, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0,5))

    # Create bounding box
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Creating rectangle around the bounding box using cvzone
            w, h = x2-x1, y2-y1

            # Mark the confidence of the object detection
            conf = math.ceil((box.conf[0]*100))/100

            # get the class names from the YOLO model
            cls_id = int(box.cls[0]) # class index
            current_class = model.names[cls_id]
            vehicles_list = ["car", "bus", "motorcycle", "truck"]            

            if current_class in vehicles_list and conf>=0.3:
                # bounding box
                # cvzone.cornerRect(img, (x1, y1, w, h), t=2, l=20, )

                # confidence
                cvzone.putTextRect(img, f'{conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2, offset=10)

                # current id'd of the vehicle
                currentArray = np.array([x1, y1, x2, y2, conf])
                # append those id's to numpy array
                detections =np.vstack((detections, currentArray))

    # maintaining the track    
    resultsTracker = tracker.update(detections)

    cv2.line(img, (line_x1, line_y), (line_x2, line_y), color=(0,0,255), thickness=3)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        cvzone.cornerRect(img, (x1, y1, w, h), t=4, l=20, rt=3)
        # cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=3, thickness=3, offset=10)

        # Find the center of the vehicles or bounding box
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (0,0,225), cv2.FILLED)

        if line_x1 < cx < line_x2 and line_y-10 < cy < line_y+10:
            if vehicle_count.count(id) == 0:
                vehicle_count.append(id)
                cv2.line(img, (line_x1, line_y), (line_x2, line_y), color=(0,255,0), thickness=3)
                

    cvzone.putTextRect(img, f'Count: {len(vehicle_count)}', (50, 50))
            
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()