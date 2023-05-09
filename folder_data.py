import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp


#path or np arrays
DATA_PATH = os.path.join('MP_Data')
#actions =  np.array(['Hello','thanks' , 'iloveyou'])
actions =  np.array(['ThankYou'])
no_sequence = 30
sequence_lenth=30

for action in actions:
    for sequence in range(no_sequence):
        #os.makedirs(os.path.join(DATA_PATH,action, str(sequence)))
        try:
            os.makedirs(os.path.join(DATA_PATH,action, str(sequence)))
        except:
            print("Not made")
            pass

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

def extract_keypoints(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose,face,lh,rh])

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5 ,min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequence):
            for frame_num in range(sequence_lenth):

                ret , frame = cap.read()
                image , results = mediapipe_detection(frame , holistic)
                #print(results)
        
                draw_landmarks(image,results)
                #show video
                if frame_num == 0:
                    image =cv2.flip(image ,1)
                    cv2.putText(image,'START COLLECTING', (120,200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) ,4 , cv2.LINE_AA)
                    cv2.putText(image,'Collecting frames for {} video number {}'.format(action ,sequence), (5,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255) ,1 , cv2.LINE_AA)
                    #cv2.imshow('openCV feed', image)
                    #print("if")
                    cv2.waitKey(3000)
                else:
                    image =cv2.flip(image ,1)
                    cv2.putText(image,'Collection frames for {} video no {}'.format(action ,sequence), (5,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255) ,1 , cv2.LINE_AA)
                    #print("elf")

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH ,action ,str(sequence),str(frame_num))
                np.save(npy_path , keypoints)

                
                cv2.imshow('openCV feed', image)
            
                if cv2.waitKey(10) & 0xFF == ord('q') :
                    break


cap.release()
cv2.destroyAllWindows()

