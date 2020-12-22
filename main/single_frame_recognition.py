import numpy
import mysql.connector
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import torch
import pickle
import os
import ctypes
import cv2

ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
workers = 0 if os.name == 'nt' else 4

detector = MTCNN(keep_all=True)

resnet = InceptionResnetV1(pretrained='vggface2').eval()

tested_image = Image.open("../images/god/1.jpg")
tested_image_embedding = resnet(detector(tested_image))

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

for x in users:
    if (tested_image_embedding - x[1]).norm().item() < 0.8:
        print(x[0], x[2])

boxes, _ = detector.detect(tested_image, )

# Draw bounding box
img_draw = tested_image.copy()
draw = ImageDraw.Draw(img_draw)
# draw.text(xy=(boxes[[0]], boxes[[1]]), text="Hello")


if boxes is not None:
    for box in boxes:
        draw.rectangle(box.tolist(), width=5)

# converting PIL image to CV format
cvImage = numpy.array(img_draw)
cvImage = cvImage[:, :, ::-1].copy()

cv2.imshow("Recognized Image", cvImage)
cv2.waitKey(0)
