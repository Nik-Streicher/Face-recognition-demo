from PIL import Image
import cv2
import sys

from utils.FaceDetector import FaceDetector, recognize_users, draw_bounding_box, convert_to_cv
from utils.MysqlConnector import MysqlConnector, encode

detector = FaceDetector()

mysql = MysqlConnector(host="localhost", user="python_user", password="password", database="python_project")
result = mysql.select_all_users()

users = encode(result=result)

v_cap = cv2.VideoCapture(0)

boxes = None
img_embedding = []
recognized_users = []
counter = 0
while True:
    success, img = v_cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    boxes, _ = detector.mtcnn.detect(img)

    detect = detector.mtcnn(img)
    if counter == 0:
        if detect is not None:
            img_embedding = detector.resnet(detect)
            recognized_users = recognize_users(tested_image_embedding=img_embedding, users=users, distance=0.9)

    image = draw_bounding_box(boxes=boxes, recognized_users=recognized_users, tested_image=img,
                              font_path='arial.ttf',
                              font_size=17)

    counter += 1
    print(counter)

    if counter == 90:
        counter = 0

    # display and close window
    cv2.imshow("test", convert_to_cv(image))
    if cv2.waitKey(1) & 0xFF == ord('g'):
        print("Program has been closed")
        sys.exit()
