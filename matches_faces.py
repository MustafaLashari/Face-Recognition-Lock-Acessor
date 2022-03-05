import time
import playsound
import face_recognition as fr
import pickle
import cv2
import os
#dlib use for cascade landmarking with 68 point x,y cordinates and make scale on face or web cam face
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(1)
face_is_match = False
face_encodings = []
known_face_encodings = pickle.load(open("encode.pickle","rb"))


while True:
        ret, frame = video_capture.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray_image,
                                             scaleFactor=1.2,
                                             minNeighbors=5,
                                             minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.imshow('input', frame)
        #cv2.waitKey(1)

        face_locations = fr.face_locations(frame, model="hog")
        face_encodings = fr.face_encodings(frame, face_locations)
               
        face_names = []
        name = "Unknown"
        cv2.imshow("Matcher", frame)
        cv2.getTickCount()
        cv2.waitKey(1)
        for face_encoding in face_encodings:
                matches = fr.compare_faces(known_face_encodings, face_encoding, 0.4)

                #find first match
                if True in matches:
                        #cv2.getTickCount()
                        first_known_face = matches.index(True)
                        print("Welcome unlock the system")
                        face_is_match = True
                        #playsound("welcome.mp3")
                        #video_capture.release()
                        #cv2.waitKey(1)
                       # time.sleep(000.1)
                else:
                        print("Access Denied")
                     #   cv2.imshow("Matcher", frame)
                     #   cv2.waitKey(1)
                      #  time.sleep(000.1)



