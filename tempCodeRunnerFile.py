import cv2
import numpy as np
import pandas as pd

WIDTH = 800
HEIGHT = 500

MAX_POINTS = 4


cap1=cv2.VideoCapture(2)
# cap2 = cv2.VideoCapture('p.mp4')
# cap3 = cv2.VideoCapture('p.mp4')
# cap4 = cv2.VideoCapture('p.mp4')



points = []

def draw_polygon(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < MAX_POINTS:
        points.append((x, y))
        print("Added point:", (x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and len(points) == MAX_POINTS:
        # Close the polygon by connecting the last point to the first one
        points.append(points[0])
        print("Closed polygon:", points)

def read_frame(cap1=None, cap2=None, cap3=None, cap4=None):
    frame1 = np.zeros((HEIGHT, WIDTH, 3), np.uint8) if cap1 is None else cap1.read()[1]
    frame1 = cv2.resize(frame1, (WIDTH, HEIGHT))

    frame2 = np.zeros((HEIGHT, WIDTH, 3), np.uint8) if cap2 is None else cap2.read()[1]
    frame2 = cv2.resize(frame2, (WIDTH, HEIGHT))

    frame3 = np.zeros((HEIGHT, WIDTH, 3), np.uint8) if cap3 is None else cap3.read()[1]
    frame3 = cv2.resize(frame3, (WIDTH, HEIGHT))

    frame4 = np.zeros((HEIGHT, WIDTH, 3), np.uint8) if cap4 is None else cap4.read()[1]
    frame4 = cv2.resize(frame4, (WIDTH, HEIGHT))

    # Build frame
    frame = build_frame(frame1, frame2, frame3, frame4)
    return frame

def build_frame(frame1, frame2, frame3, frame4):
    frame11 = np.hstack((frame1, frame2))
    frame22 = np.hstack((frame3, frame4))

    frame = np.vstack((frame11, frame22))

    return frame

if __name__ == "__main__":
    while True:
        cv2.namedWindow('Draw Polygon')
        cv2.setMouseCallback('Draw Polygon', draw_polygon)

        
        frame = read_frame(cap1)
        # Draw the polygon on the frame
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)

        cv2.imshow('Draw Polygon', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break