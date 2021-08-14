import argparse

import numpy as np
import cv2

import torch
import torchvision

from models.pfld import PFLDInference, AuxiliaryNet
from Facedetector.mtcnn.detector import mtcnn_detect
from Facedetector.yolov5.detector import yolo_detect
from Facedetector.pyramidbox.detector import pyramidbox_detect

from scipy.spatial import distance as dist
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EAR计算
def eye_aspect_ratio(eye):
    # 计算距离，竖直的
    A = dist.euclidean(eye[1], eye[7])
    B = dist.euclidean(eye[3], eye[5])
    D = dist.euclidean(eye[2], eye[6])
    # 计算距离，水平的
    C = dist.euclidean(eye[0], eye[4])
    # ear值
    ear = (A + B + D) / (3.0 * C)
    return ear


def main(args):
    #选择face detector
    if args.face_detector == "mtcnn":
        face_detector = mtcnn_detect
    elif args.face_detector == 'yolo':
        face_detector = yolo_detect
    elif args.face_detector == 'pyramidbox':
        face_detector = pyramidbox_detect

    # 98个特征点标序
    FACIAL_LANDMARKS_98_IDXS = OrderedDict([
        ("mouth", (76, 97)),
        ("right_eyebrow", (33, 42)),
        ("left_eyebrow", (42, 51)),
        ("right_eye", (60, 68)),
        ("left_eye", (68, 76)),
        ("nose", (51, 60)),
        ("jaw", (0, 33))
    ])

    # 设置判断参数
    EYE_AR_THRESH = 0.32
    EYE_AR_CONSEC_FRAMES = 3

    # 初始化计数器
    COUNTER = 0
    TOTAL = 0

    # 分别取两个眼睛区域
    (lStart, lEnd) = FACIAL_LANDMARKS_98_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_98_IDXS["right_eye"]

    checkpoint = torch.load(args.model_path, map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
    pfld_backbone.eval()
    pfld_backbone = pfld_backbone.to(device)
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])

    cap = cv2.VideoCapture(0)

    width = 1200
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    r = width / float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (width, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * r))

    out = cv2.VideoWriter('camera_test2.mp4', fourcc, fps, size)

    while True:
        ret, img = cap.read()
        if not ret: break
        height, width = img.shape[:2]
        bounding_boxes, _ = face_detector(img)
        for box in bounding_boxes:
            x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)

            w = x2 - x1 + 1
            h = y2 - y1 + 1
            cx = x1 + w // 2
            cy = y1 + h // 2

            size = int(max([w, h]) * 1.1)
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            edx1 = max(0, -x1)
            edy1 = max(0, -y1)
            edx2 = max(0, x2 - width)
            edy2 = max(0, y2 - height)

            cropped = img[y1:y2, x1:x2]
            if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
                cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2,
                                             cv2.BORDER_CONSTANT, 0)

            input = cv2.resize(cropped, (112, 112))
            input = transform(input).unsqueeze(0).to(device)
            _, landmarks = pfld_backbone(input)
            pre_landmark = landmarks[0]
            pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size, size] - [edx1, edy1]
            pre_landmark = np.float32(pre_landmark)

            # 分别计算ear值
            leftEye = pre_landmark[lStart:lEnd]
            rightEye = pre_landmark[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # 算一个平均的
            ear = (leftEAR + rightEAR) / 2.0

            # 检查是否满足阈值
            if ear < EYE_AR_THRESH:
                COUNTER += 1

            else:
                # 如果连续几帧都是闭眼的，总数算一次
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1

                # 重置
                COUNTER = 0

            # 绘制眼睛区域
            leftEyeHull = cv2.convexHull(leftEye) + (x1, y1)
            rightEyeHull = cv2.convexHull(rightEye) + (x1, y1)
            cv2.drawContours(img, [leftEyeHull.astype(int)], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull.astype(int)], -1, (0, 255, 0), 1)

            #for (x, y) in pre_landmark.astype(np.int32):
                #cv2.circle(img, (x1 + x, y1 + y), 1, (0, 0, 255))

            # 显示
            cv2.putText(img, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('face_landmark_98', img)
        if cv2.waitKey(10) == 27:
            break

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path',
                        default="./checkpoint/snapshot/checkpoint.pth.tar",
                        type=str)
    parser.add_argument("-f", "--face_detector", type=str, default="pyramidbox",
                    help="choose face detector among mtcnn, yolo and pyramidbox")
    parser.add_argument("-l", "--landmark_detector", type=str, default="mobileNet",
                    help="choose face detector among ...")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
