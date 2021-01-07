import numpy
import torch
from PIL import ImageFont, ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1
import ctypes

ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')


# return list of recognized users in format -> [name, access]
# "distance" defines the distance between the recognized faces.
def recognize_users(face_embedding, users, distance):
    recognized_users = []

    for y in face_embedding:
        flag = True
        recognized_user = []
        for x in users:
            if (y - x[1]).norm().item() < distance:
                print(x[0], x[2])
                recognized_user = [x[0], str(x[2])]
                flag = False

        if flag:
            recognized_users.append(["unknown", "None"])
        else:
            recognized_users.append(recognized_user)

    print(recognized_users)
    return recognized_users


# draw bounding box, name and access status
def draw_bounding_box(boxes, recognized_users, pil_image, font_path, font_size):
    img_draw = pil_image.copy()
    font_type = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(img_draw)

    counter = 0
    if boxes is not None:
        for box in boxes:
            a, b, c, d = box
            draw.rectangle(box.tolist(), width=3)

            if (counter + 1) <= len(recognized_users):
                draw.text((a + 5, b + 3), recognized_users[counter][0], font=font_type)
                draw.text((c - 43, d - 20), recognized_users[counter][1], font=font_type)
                counter += 1

    return img_draw


# converts PIL image to CV format
def convert_to_cv(pil_image):
    cvImage = numpy.array(pil_image)
    return cvImage[:, :, ::-1].copy()


class FaceDetector:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
