
import cv2
from cvzone.HandTrackingModule import HandDetector
import tensorflow
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
from tensorflow.keras.models import load_model


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("trainedModel.h5","labels.txt")
# Classifier = load_model("A_B_c_sign_language_model.h5")
offset = 40
imgSize = 400
counter = 0
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  



while True:
    success, img = cap.read()
    imgOutput =  img.copy()
    hands, img = detector.findHands(img)

    if hands:
        if len(hands) == 1:
            # If only one hand is present, take the bounding box around that hand
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        else:
            # If two hands are present, calculate a common bounding box
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
            prediction , index = classifier.getPrediction(imgWhite)
            print(prediction,index)
            
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal )/2)
            imgWhite[hGap:hCal + hGap,:] = imgResize
            prediction , index = classifier.getPrediction(imgWhite)
            print(prediction,index)

        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)

        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow("imageWhite",imgWhite)
    cv2.imshow("image", imgOutput)
    cv2.waitKey(1)