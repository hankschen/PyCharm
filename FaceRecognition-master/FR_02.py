#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2
import pickle
from TextToSpeech import TextToSpeech


def getName(model, y, names, img):
    # im = cv2.imread("./user3.pgm", cv2.IMREAD_GRAYSCALE)
    [p_label, p_confidence] = model.predict(img)
    # print "Predicted label = %d %s (confidence=%.2f)" % (p_label, names[y.tolist().index(p_label)], p_confidence)
    return (names[y.tolist().index(p_label)], p_confidence)


model = cv2.createEigenFaceRecognizer()
model.load("faces.yml")
with open("names.txt", 'rb') as f:
    names = pickle.load(f)
with open("y.txt", 'rb') as f:
    indices = pickle.load(f)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
inx = 0

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        head = gray[y:y + h, x:x + w]
        # head = cv2.resize(head, (92/ w, 112/h), interpolation=cv2.INTER_CUBIC)
        sX = 92.0 / w
        sY = 112.0 / w
        head = cv2.resize(head, None, fx=sX, fy=sY, interpolation=cv2.INTER_CUBIC)
        # head = cv2.equalizeHist(head)
        cv2.imshow('Clip', head)
        cv2.moveWindow('Clip', 700, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        name, conf = getName(model, indices, names, head)
        if conf < 5000:
            cv2.putText(frame, str(names.index(name)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        else:
            cv2.putText(frame, "???", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            # print "X: %d Y: %d W: %d, H: %d" % (x, y, w, h)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    cv2.moveWindow('Video', 0, 0)
    
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s'):
        fileName = './FaceCaptured/' + str(inx) + '.pgm'
        print fileName
        cv2.imwrite(fileName, head)
        inx += 1
    elif k == ord('i'):
        name, conf = getName(model, indices, names, head)
        print " %s %.2f " % (name, conf)
        strToSay = u"你好嗎" + name
        TextToSpeech.saySomthing(strToSay, "zh-tw")
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
