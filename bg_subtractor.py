import cv2
import numpy as np

cap = cv2.VideoCapture('video2.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

algo = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    try:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 5)
        img_sub = algo.apply(blur)
        # dilat = cv2.dilate(img_sub, np.ones((7, 7)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilatda = cv2.morphologyEx(img_sub, cv2.MORPH_CLOSE, kernel)
        dilatda = cv2.morphologyEx(dilatda, cv2.MORPH_CLOSE, kernel)
        contours, h = cv2.findContours(dilatda, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)

        for (i,c) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(c)
            validate_counter = (w >= 80) and (h >= 80)
            # if cv2.contourArea(c) < 900:
            #     continue
            if not validate_counter or cv2.contourArea(c) < 1500:
                continue
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Detector', dilatda)
        cv2.imshow('frame', frame)
    except Exception as e:
        print(f"Error during processing: {e}")
        break

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()