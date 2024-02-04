import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time 

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 40
imgSize = 400
counter = 0

folder = "data\\Z"
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        if len(hands) == 1:
            
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
        else:
            
            x1, y1, w1, h1 = hands[0]['bbox']
            x2, y2, w2, h2 = hands[1]['bbox']

            x = min(x1, x2) - offset
            y = min(y1, y2) - offset
            w = max(x1 + w1, x2 + w2) + offset - x
            h = max(y1 + h1, y2 + h2) + offset - y

            imgCrop = img[y:y + h, x:x + w]
            imgCropShape = imgCrop.shape

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255

        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal )/2)
            imgWhite[:,wGap:wCal + wGap] = imgResize

        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal )/2)
            imgWhite[hGap:hCal + hGap,:] = imgResize


        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow("imageWhite",imgWhite)
    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter +=1
        cv2.imwrite(f"{folder}//Image_{time.time()}.png",imgWhite)
        print(counter)