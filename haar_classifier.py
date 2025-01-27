import cv2
vehiclexml=cv2.CascadeClassifier('vehicle.xml')

def detection(frame):
    vehicle=vehiclexml.detectMultiScale(frame,1.15, 3)
    for (x,y,w,h) in vehicle:
        cv2.rectangle(frame,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)

    return frame
def capturescreen():
    realtimevideo=cv2.VideoCapture('video2.mp4')
    while realtimevideo.isOpened():
        ret,frame=realtimevideo.read()
        controlkey=cv2.waitKey(1)
        if ret:
            vehicleframe=detection(frame)
            cv2.imshow('vehicle detection',vehicleframe)
        else:
            break
        if controlkey==ord('q'):
            break


    realtimevideo.release()
    cv2.destroyAllWindows()

capturescreen()

# cap = cv2.VideoCapture('video2.mp4')

# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break