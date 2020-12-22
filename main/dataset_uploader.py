import mysql.connector
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
import pickle
import os
import cv2
import ctypes

ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
workers = 0 if os.name == 'nt' else 4

database = mysql.connector.connect(
    host="localhost",
    user="python_user",
    password="password",
    database="python_project"
)

image = cv2.imread("")
img = Image.open("")

mtcnn = MTCNN(keep_all=True)

resnet = InceptionResnetV1(pretrained='vggface2').eval()

cursor = database.cursor()


def collate_fn(x):
    return x[0]


dataset = datasets.ImageFolder('../images')
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
users = []

for x, y in loader:
    x_aligned = mtcnn(x)
    if x_aligned is not None:
        users.append([dataset.idx_to_class[y], resnet(x_aligned)])

for x in users:
    sql = "INSERT INTO users (name_surname, embedding, access) VALUES (%s, %s, %s)"
    val = (x[0], pickle.dumps(x[1]).decode('ISO-8859-1'), True)

    cursor.execute(sql, val)

database.commit()

