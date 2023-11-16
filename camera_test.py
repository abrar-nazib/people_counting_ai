import cv2

CAM_NUM_1 = 2
CAM_NUM_2 = 4
cap1 = cv2.VideoCapture(CAM_NUM_1)
cap2 = cv2.VideoCapture(CAM_NUM_2)


while True:
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()
    cv2.imshow("frame1", frame1)
    cv2.imshow("frame2", frame2)
    if cv2.waitKey(1) == ord('q'):
        break
cap1.release()
# cap2.release()
cv2.destroyAllWindows()