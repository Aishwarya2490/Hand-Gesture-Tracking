import cv2
import mediapipe as mp
import numpy as np
import time

class handdetector():
    def __init__(self,mode=False,maxHands = 2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpdraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgrgb)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
               if draw:
                   self.mpdraw.draw_landmarks(img, handlms, self.mphands.HAND_CONNECTIONS)

        return img
    def findposition(self,img,handno=0,draw = True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(myhand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

        return lm_list

def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector = handdetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lm_list = detector.findposition(img)
        if len(lm_list) !=0:
            print(lm_list[4])
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (18, 78), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        cv2.imshow('Image',img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()