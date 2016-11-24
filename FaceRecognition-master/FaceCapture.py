import cv2

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
        cv2.imshow('Clip', head)
        cv2.moveWindow('Clip', 700, 0)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # print "X: %d Y: %d W: %d, H: %d" % (x, y, w, h)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    # head = cv2.equalizeHist(head)
    cv2.moveWindow('Video', 0, 0)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s'):
        fileName = './faces/' + str(inx) + '.pgm'
        print fileName
        cv2.imwrite(fileName, head)
        inx += 1

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
