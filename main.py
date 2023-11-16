import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone
import numpy as np
import time

model=YOLO('yolov8n.pt')

tracker = Tracker()

WIDTH = 800
HEIGHT = 500


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap1=cv2.VideoCapture(2)
# cap2 = cv2.VideoCapture('p.mp4')
# cap3 = cv2.VideoCapture('p.mp4')
# cap4 = cv2.VideoCapture('p.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

Y_OFFSET = 30
area1=[(260, 303), (351, 313), (346, 350), (228, 326), (260, 303)]
area2= [(220, 335), (348, 356), (337, 379), (211, 356), (220, 335)]


entry = 0
exit = 0
area1_dict = {}
area2_dict = {}

while True:    
    ret,frame1 = cap1.read()
    # ret, frame2 = cap2.read()
    # ret, frame3 = cap3.read()
    # ret, frame4 = cap4.read()
    t1 = time.time()
    if not ret:
        break


#    count += 1
#    if count % 3 != 0:
#        continue
    frame1 = cv2.resize(frame1,(WIDTH, HEIGHT))
    # frame2 = cv2.resize(frame2, (1020, 500))
    frame2 = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    # frame3 = cv2.resize(frame3, (1020, 500))
    frame3 = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    # frame4 = cv2.resize(frame4, (1020, 500))
    frame4 = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    
    # Connect the two frames
    frame11 = np.hstack((frame1, frame2))
    frame22 = np.hstack((frame3, frame4))

    frame = np.vstack((frame11, frame22))

    # Inference model on photo
    results=model.predict(frame, verbose=False)
    
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    
    bbox_list = []
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        if 'person' in c:
            bbox_list.append([x1,y1,x2,y2])
    bbox_idx = tracker.update(bbox_list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox

        # Position of the bottom circle        
        cx,cy = int((x3 + x4) / 2), y4
        
        # Check if the person is in the area of interest
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.putText(frame, str(id), (x3, y3 - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)

        # Check if the person is in the area of interest
        area1_in = cv2.pointPolygonTest(np.array(area1), (cx, cy), False)
        area2_in = cv2.pointPolygonTest(np.array(area2), (cx, cy), False)

        if(area1_in > 0):
            # Check whether the person is already in the list
            if id not in area1_dict:
                area1_dict[id] = 1

            # Check whether the person was in area2 before
            if id in area2_dict:
                exit += 1
                del area2_dict[id]

        if(area2_in > 0):
            # Check whether the person is already in the list
            if id not in area2_dict:
                area2_dict[id] = 1

            # Check whether the person was in area1 before
            if id in area1_dict:
                entry += 1
                del area1_dict[id]

            
   
    # Draw the area of interest
    cv2.polylines(frame, [np.array(area1)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(area2)], True, (0, 255, 0), 2)

    # Display the entry and exit count
    cv2.putText(frame, "Entry: {}".format(str(entry)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Exit: {}".format(str(exit)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    t2 = time.time()
    fps = 1 / (t2 - t1)
    cv2.putText(frame, "FPS: {}".format(str(round(fps, 2))), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap1.release()
cv2.destroyAllWindows()

