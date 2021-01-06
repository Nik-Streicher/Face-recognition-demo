import numpy
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import sys
import pickle
import ctypes
import mysql.connector
from numba import njit

ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

database = mysql.connector.connect(
    host="localhost",
    user="python_user",
    password="password",
    database="python_project"
)

cursor = database.cursor()

cursor.execute("SELECT * FROM users")
result = cursor.fetchall()

users = [[num[0], pickle.loads(num[1].encode('ISO-8859-1')), num[2] == 1] for num in result]
recognized_users = []


resnet = InceptionResnetV1(pretrained='vggface2').eval()
detector = MTCNN(keep_all=True, device='cuda:0')
v_cap = cv2.VideoCapture(0)
my_image = Image.open("owner.jpg")

boxes = None
counter = 0

while True:
    success, img = v_cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    boxes, _ = detector.detect(img)

    # if counter == 0:
    #     img_embedding = resnet(detector(img))
    #     for y in img_embedding:
    #         flag = True
    #         recognized_user = []
    #         for x in users:
    #             if (y - x[1]).norm().item() < 0.9:
    #                 print(x[0], x[2])
    #                 recognized_user = [x[0], str(x[2])]
    #                 flag = False
    #
    #         if flag:
    #             recognized_users.append(["unknown", "None"])
    #         else:
    #             recognized_users.append(recognized_user)

    counter += 1

    if counter == 90:
        counter = 0

    print(counter)
    # Draw bounding box
    img_draw = img.copy()
    font_type = ImageFont.truetype('arial.ttf', 17)
    draw = ImageDraw.Draw(img_draw)

    in_counter = 0
    if boxes is not None:
        for box in boxes:
            a, b, c, d = box
            draw.rectangle(box.tolist(), width=3)
            # draw.text((a + 5, b + 3), recognized_users[in_counter][0], font=font_type)
            # draw.text((c - 43, d - 20), recognized_users[in_counter][1], font=font_type)
            in_counter += 1

    # converting PIL image to CV format
    cvImage = numpy.array(img_draw)
    cvImage = cvImage[:, :, ::-1].copy()

    # display and close window
    cv2.imshow("test", cvImage)
    if cv2.waitKey(1) & 0xFF == ord('g'):
        print("Program has been closed")
        sys.exit()
