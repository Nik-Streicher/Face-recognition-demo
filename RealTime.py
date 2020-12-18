import numpy
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import sys
import ctypes
from numba import njit

ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

resnet = InceptionResnetV1(pretrained='vggface2').eval()
detector = MTCNN(keep_all=True, device='cuda:0')
v_cap = cv2.VideoCapture(0)
my_image = Image.open("owner.jpg")

while True:
    success, img = v_cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # img_embedding = resnet(detector(img))
    # if (img_embedding - resnet(detector(my_image))).norm().item() < 0.8:
    #     print("we got him")
    # else:
    #     print("we got a spy here")
    # print(img_embedding)

    boxes, _ = detector.detect(img)

    # Draw bounding box
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    if boxes is not None:
        for box in boxes:
            draw.rectangle(box.tolist(), width=5)

    # converting PIL image to CV format
    cvImage = numpy.array(img_draw)
    cvImage = cvImage[:, :, ::-1].copy()

    cv2.imshow("test", cvImage)
    if cv2.waitKey(1) & 0xFF == ord('g'):
        print("Program has been closed")
        sys.exit()
