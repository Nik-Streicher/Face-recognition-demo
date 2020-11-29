import cv2

capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("D:\pythonProject\haarcascade_frontalface_default.xml")

while True:
    success, image = capture.read()

    faces = faceCascade.detectMultiScale(image, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Video", image)
    if cv2.waitKey(1) & 0xFF == ord('g'):
        break

