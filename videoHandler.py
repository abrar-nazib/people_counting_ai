import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tracker import*

CAMERA_ARRAY = [2, 4, 6, 8]

CAM1_AREA_UP =  [(214, 388), (194, 414), (310, 426), (316, 404), (214, 388)]
CAM1_AREA_DOWN =   [(186, 421), (176, 436), (301, 445), (306, 436), (186, 421)]

CAM2_AREA_UP =  [(1019, 391), (997, 419), (1112, 420), (1116, 398), (1019, 391)]
CAM2_AREA_DOWN =   [(983, 423), (969, 436), (1105, 446), (1108, 427), (983, 423)]

CAM3_AREA_UP = [(954, 708), (960, 739), (1298, 746), (1297, 716), (954, 708)]
CAM3_AREA_DOWN =  [(961, 755), (961, 786), (1305, 798), (1298, 759), (961, 755)]


WIDTH = 800
HEIGHT = 500

MAX_POINTS = 4




polygons = []
points = []


def draw_polygon(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < MAX_POINTS:
        points.append((x, y))
        print("Added point:", (x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and len(points) == MAX_POINTS:
        # Close the polygon by connecting the last point to the first one
        points.append(points[0])
        print("Closed polygon:\n", points)
        polygons.append(points)
        points = []


class Frame:
    def __init__(self, serial, cap=None):
        self.cap = cap
        self.serial = serial
        if self.cap is not None:
            self.frame = self.cap.read()[1]
            self.frame = cv2.resize(self.frame, (WIDTH, HEIGHT))
        else:
            self.frame = np.ones((HEIGHT, WIDTH, 3), np.uint8) * 100
        self.area_up = [(144, 437), (395, 432), (385, 351), (145, 360), (144, 437)]
        self.area_down =  [(149, 529), (414, 523), (399, 450), (147, 451), (149, 529)]
        self.ids_area_up = {}
        self.ids_area_down = {}
        self.entry = 0
        self.exit = 0

        # Set the serial text position on the screen
        self.cam_serial_pos = (20 , 20)
        self.enter_text_pos = (20 , 40)
        self.exit_text_pos = (20 , 60 )



    def update_frame(self):
        if self.cap is not None:
            self.frame = self.cap.read()[1]
            self.frame = cv2.resize(self.frame, (WIDTH, HEIGHT))
        else:
            self.frame = cv2.resize(self.frame, (WIDTH, HEIGHT))
        self.draw_info()

    def draw_text(self, text, pos):
        self.frame = cv2.putText(self.frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    def draw_info(self):
        self.draw_text("Camera {}".format(self.serial), self.cam_serial_pos)
        self.draw_text("Entry: {}".format(self.entry), self.enter_text_pos)
        self.draw_text("Exit: {}".format(self.exit), self.exit_text_pos)

    def set_polygons(self, area_up, area_down):
        # Global polygons
        self.area_up = area_up
        self.area_down = area_down

    def increment_entry(self):
        self.entry += 1
    
    def increment_exit(self):
        self.exit += 1




class HumanDetector:
    def __init__(self, model):
        print("HumanDetector initialized")
        self.model = YOLO(model)
        self.conf = open("coco.txt", "r")
        self.data = self.conf.read()
        self.class_list = self.data.split("\n")
        self.tracker = Tracker()

    def detect(self, frame):
        results=self.model.predict(frame, verbose=False)
    
        a=results[0].boxes.data
        px=pd.DataFrame(a).astype("float")
    
        bbox_list = []
        for index,row in px.iterrows():
    #        print(row)
    
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])
            
            c=self.class_list[d]
            if 'person' in c:
                bbox_list.append([x1,y1,x2,y2])

        return bbox_list

    def track(self, bbox_list):
        bbox_idx = self.tracker.update(bbox_list)
        return bbox_idx

class VideoFeed:
    def __init__(self, camera_array):
        self.camera_array = camera_array
        self.cap_array = []
        cap0 = cv2.VideoCapture(camera_array[0])
        # cap1 = None
        cap1 = cv2.VideoCapture(camera_array[1])
        cap2 = None
        cap3 = None
        self.cap_array.append(cap0)
        self.cap_array.append(cap1)
        self.cap_array.append(cap2)
        self.cap_array.append(cap3)
        self.human_detector = HumanDetector("yolov8n.pt")

        self.frames = []
        frame1 = Frame(0)
        frame1.set_polygons(CAM1_AREA_UP, CAM1_AREA_DOWN)

        frame2 = Frame(1, cap0)
        frame2.set_polygons(CAM2_AREA_UP, CAM2_AREA_DOWN)
        # frame2 = Frame(1)
        frame3 = Frame(2)
        frame3.set_polygons(CAM3_AREA_UP, CAM3_AREA_DOWN)
        
        frame4 = Frame(3)
        frame4.set_polygons(CAM3_AREA_UP, CAM3_AREA_DOWN)

        self.frames.append(frame1)
        self.frames.append(frame2)
        self.frames.append(frame3)
        self.frames.append(frame4)

        self.stacked_frame = np.zeros((HEIGHT*2, WIDTH*2, 3), np.uint8)


    def draw_polygons_on_stacked_frame(self):
        for frame in self.frames:
            self.stacked_frame = cv2.polylines(self.stacked_frame, np.array([frame.area_up]), True, (0, 255, 0), 2)
            self.stacked_frame = cv2.polylines(self.stacked_frame, np.array([frame.area_down]), True, (255, 0, 0), 2)

    def stack_frames(self):
        frame1 = np.hstack((self.frames[0].frame, self.frames[1].frame))
        frame2 = np.hstack((self.frames[2].frame, self.frames[3].frame))
        self.stacked_frame = np.vstack((frame1, frame2))
        

    def update_frames(self):
        for frame in self.frames:
            frame.update_frame()

    def update_stacked_frame(self):
        self.update_frames()
        self.stack_frames()
        self.draw_polygons_on_stacked_frame()

    def get_stacked_frame(self):
        return self.stacked_frame

    def show_stacked_frame(self):
        cv2.imshow('Stacked Camera Feed', self.stacked_frame)
    
    def check_point_in_polygon(self, point, polygon):
        result = cv2.pointPolygonTest(np.array(polygon), point, False)
        return result >= 0
    
    def show_bbox(self,track_bbox_array):
        for bbox in track_bbox_array:
            x1, y1, x2, y2, id = bbox
            cv2.rectangle(self.stacked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.stacked_frame, str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Put circle in the bottm center of the bbox
            cx,cy = int((x1 + x2) / 2), y2
            cv2.circle(self.stacked_frame, (cx, cy), 5, (0, 0, 0), -1)

    def check_area_crossed(self):
        detect_bbox_list = self.human_detector.detect(self.stacked_frame)
        tracked_bbox_list = self.human_detector.track(detect_bbox_list)
        self.show_bbox(tracked_bbox_list)
        for frame in self.frames:
            for bbox in tracked_bbox_list:
                x3, y3, x4, y4, id = bbox
                cx,cy = int((x3 + x4) / 2), y4
                if self.check_point_in_polygon((cx, cy), frame.area_up):
                    if id not in frame.ids_area_up:
                        frame.ids_area_up[id] = 1
                    if id in frame.ids_area_down:
                        del frame.ids_area_down[id]
                        frame.increment_entry()
                
                if self.check_point_in_polygon((cx, cy), frame.area_down):
                    if id not in frame.ids_area_down:
                        frame.ids_area_down[id] = 1
                
                    if id in frame.ids_area_up:
                        del frame.ids_area_up[id]
                        frame.increment_exit()




if __name__ == "__main__":
    video_feed = VideoFeed(CAMERA_ARRAY)
    while True:
        cv2.namedWindow('Draw Polygon')
        cv2.setMouseCallback('Draw Polygon', draw_polygon)

        
        video_feed.update_stacked_frame()
        frame = video_feed.get_stacked_frame()
        video_feed.check_area_crossed()
        # Draw the polygons on the frame
        for polygon in polygons:
            cv2.polylines(frame, np.array([polygon]), True, (0, 255, 0), 2)
        
        for point in points:
            # Draw circles on the points
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

        cv2.imshow('Draw Polygon', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    