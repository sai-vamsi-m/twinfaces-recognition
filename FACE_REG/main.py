import cv2 
import face_recognition 
import numpy as np

ingElon = face_recognition.load_image_file("ImageBasic/Elon Musk.jpg")
imgElon = cv2.cvtColor(ingElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("ImageBasic/Elon Test.jpg") 
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
faceEncode = face_recognition.face_encodings(imgElon)[0]

cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 0), 3)

faceLocTest = face_recognition.face_locations(imgTest)[0]
faceEncodeTest = face_recognition.face_encodings(imgTest)[0]

result = face_recognition.compare_faces([faceEncode], faceEncodeTest)
faceDis = face_recognition.face_distance([faceEncode], faceEncodeTest)
print(result, faceDis)

cv2.putText(imgTest, f'{result} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)
