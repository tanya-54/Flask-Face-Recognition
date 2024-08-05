from ast import Return
import numpy as np
import face_recognition as fr
import cv2


face_cap= cv2.CascadeClassifier("C:/Users/Dell/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)


# it will return a no of arrays 
tanu_image = fr.load_image_file("Tanu/tanu.jpg")
tanu_face_encoding= fr.face_encodings(tanu_image)[0]

known_face_encodings = [tanu_face_encoding]
known_face_names =["Tanya"]

while True:
    ret, frame = video_capture.read()  # takes frame of the camera 

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # changes the color of frame 

    face_locations = fr.face_locations(rgb_frame)  # checks where are the faces in this picture  
    face_encodings = fr.face_encodings(rgb_frame, face_locations)  # then encoded


    for ( top , right ,bottom ,left) , face_encoding in zip(face_locations , face_encodings):

        matches = fr.compare_faces(known_face_encodings , face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encodings , face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]



        cv2.rectangle(frame , (left , top), (right , bottom) , (0,0,255), 2)

        cv2.rectangle(frame , (left, bottom -35), (right, bottom), (0, 0, 255),cv2.FILLED) 
        font =  cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left +6 , bottom -6), font, 1.0, (255,255,255),1 )
        
    cv2.imshow('webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()

cv2.destroyAllWindows()






