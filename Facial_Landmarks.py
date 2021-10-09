import cv2
import numpy as np
import dlib
import glob
import face_utils

predictor_path = "shape_predictor_68_face_landmarks.dat"
cap = cv2.VideoCapture("v1.MOV")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
findex = -1

while (True):
    _,frame = cap.read()

    dets = detector(frame, 1)
    for k, d in enumerate(dets):
        shape = predictor(frame, d) 

    findex += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(gray)
    for k, d in enumerate(dets):
        shape = predictor(gray, d)
        shape1 = face_utils.shape_to_np(shape)        
    
    for (x, y) in shape1:
	    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    cv2.putText(frame, "FIndex:  " + str(findex), (20, 190), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0,0), 1)
    cv2.imshow("frame", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        cap.release()
        break
        
        
cv2.destroyAllWindows()
    

