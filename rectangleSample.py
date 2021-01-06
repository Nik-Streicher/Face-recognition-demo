import numpy
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2

detector = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

v_cap = cv2.VideoCapture(0)
success, img = v_cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img)

img_embedding = resnet(detector(img))
print(img_embedding)

boxes, ignored, points = detector.detect(img, landmarks=True)

# Draw bounding box
img_draw = img.copy()
draw = ImageDraw.Draw(img_draw)

for i, (box, point) in enumerate(zip(boxes, points)):
    draw.rectangle(box.tolist(), width=5)

# converting PIL image to CV format
cvImage = numpy.array(img_draw)
cvImage = cvImage[:, :, ::-1].copy()

cv2.imshow("test", cvImage)
cv2.waitKey(0)
