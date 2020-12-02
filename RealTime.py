import numpy
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import cv2

detector = MTCNN(keep_all=True)
v_cap = cv2.VideoCapture(0)

while True:
    success, img = v_cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    boxes, ignored, points = detector.detect(img, landmarks=True)

    # Draw bounding box
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    try:
        for i, (box, point) in enumerate(zip(boxes, points)):
            draw.rectangle(box.tolist(), width=5)
    except TypeError:
        pass

    # converting PIL image to CV format
    cvImage = numpy.array(img_draw)
    cvImage = cvImage[:, :, ::-1].copy()

    cv2.imshow("test", cvImage)
    if cv2.waitKey(1) & 0xFF == ord('g'):
        break
