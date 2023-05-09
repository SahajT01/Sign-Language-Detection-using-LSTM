import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp

#
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
#

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
    return image , results

def draw_landmarks(image , results):
    mp_drawing.draw_landmarks(image,results.face_landmarks , mp_holistic.FACEMESH_CONTOURS, mp_drawing.DrawingSpec( color=(80,110,10), thickness=1 ,circle_radius=1), mp_drawing.DrawingSpec( color=(80,256,121), thickness=1 ,circle_radius=1))
    #mp_drawing.draw_landmarks(image,results.face_landmarks , mp_holistic.CONTOURS, mp_drawing.DrawingSpec( color=(80,110,10), thickness=1 ,circle_radius=1), mp_drawing.DrawingSpec( color=(80,256,121), thickness=1 ,circle_radius=1))
    mp_drawing.draw_landmarks(image,results.pose_landmarks , mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec( color=(80,22,10), thickness=2 ,circle_radius=4), mp_drawing.DrawingSpec( color=(80,44,121), thickness=2 ,circle_radius=2))
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks , mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec( color=(121,11,250), thickness=2 ,circle_radius=4), mp_drawing.DrawingSpec( color=(121,89,76), thickness=2 ,circle_radius=2))
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks , mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec( color=(245,117,66), thickness=2 ,circle_radius=4), mp_drawing.DrawingSpec( color=(245,56,210), thickness=2 ,circle_radius=2))

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5 ,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        #read video
        ret , frame = cap.read()
        
    
        image , results = mediapipe_detection(frame , holistic)
        #print(results)

        draw_landmarks(image,results)
        #show video
        image =cv2.flip(image ,1)
        cv2.imshow('openCV feed', image)
    
        if cv2.waitKey(10) & 0xFF == ord('q') :
            break
cap.release()
cv2.destroyAllWindows()