import cv2

cv2.namedWindow("frame") #, cv2.WINDOW_NORMAL or cv2.WINDOW_FREERATIO
cap = cv2.VideoCapture(1)
height, width = 720, 1280
cap.set(cv2.CAP_PROP_FRAME_WIDTH ,width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cap.set(cv2.CAP_PROP_FPS, 60)

while(True):
    success, frame=cap.read()
    cv2.imshow("frame", frame)
    ckey = cv2.waitKey(10)
    print("press key:",ckey)
    if(ckey == ord('c')):        
        break

#
