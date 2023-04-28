import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch

from tqdm import tqdm


class loader(Dataset):

    def __init__(self, data_path, data_type, leave_out):
        self.lines = []
        self.labels = {}
        self.data_path = data_path
        self.data_type = data_type
        label_path = data_path + "\\Label"
        subjects = os.listdir(label_path)
        subjects.sort()
        for subject in tqdm(subjects):
            if data_type == "train" and subject == leave_out + ".label":
                # print("Leave out " + leave_out)
                continue
            elif data_type == "test" and subject != leave_out + ".label":
                continue
            with open(os.path.join(label_path, subject), "r") as file:
                for line in file:
                    #   0        1       2        3        4        5        6       7       8        9       10         11             12
                    # faceImg leftImg rightImg gridImg originImg whicheye 2DPoint Headrot HeadTrans ratio FaceCorner LeftEyeCorner RightEyeCorner
                    #

                    labels = line.strip().split(' ')
                    if labels[0] == "Face":  # skip first line
                        continue

                    labels = [i.split(',') for i in labels]

                    faceCorner = [float(l) for l in labels[10]]
                    faceWidth = faceCorner[2] - faceCorner[0]
                    faceHeight = faceCorner[3] - faceCorner[1]

                    leftEyeCorner = [float(l) for l in labels[11]]
                    leftEyeWidth = leftEyeCorner[2] - leftEyeCorner[0]
                    leftEyeHeight = leftEyeCorner[3] - leftEyeCorner[1]

                    rightEyeCorner = [float(l) for l in labels[12]]
                    rightEyeWidth = rightEyeCorner[2] - rightEyeCorner[0]
                    rightEyeHeight = rightEyeCorner[3] - rightEyeCorner[1]

                    label = [float(l) for l in line.strip().split(' ')[6].split(",")]

                    # bbox: x, y of upper left corner, width, height
                    # rects: [rightEye, face, leftEye]
                    rects = [rightEyeCorner[0], rightEyeCorner[1], rightEyeWidth, rightEyeHeight, faceCorner[0],
                             faceCorner[1], faceWidth, faceHeight, leftEyeCorner[0], leftEyeCorner[1], leftEyeWidth,
                             leftEyeHeight]
                    temp = [labels[0][0].split("\\")[0], labels[0][0], rects, label, labels[1][0], labels[2][0]]
                    self.lines.append(temp)


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        #   0        1         2      3       4         5
        # subject, facePath, rects, label, leftPath, rightPath
        #
        line = self.lines[idx]

        Image_path = os.path.join(self.data_path, 'Image')

        face_img = cv2.imread(os.path.join(Image_path, line[1]))
        leftEye_img = cv2.imread(os.path.join(Image_path, line[4]))
        rightEye_img = cv2.imread(os.path.join(Image_path, line[5]))

        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img / 255
        face_img = face_img.transpose(2, 0, 1)

        leftEye_img = cv2.resize(leftEye_img, (112, 112))
        leftEye_img = cv2.cvtColor(leftEye_img, cv2.COLOR_BGR2RGB)
        leftEye_img = leftEye_img / 255
        leftEye_img = leftEye_img.transpose(2, 0, 1)

        rightEye_img = cv2.resize(rightEye_img, (112, 112))
        rightEye_img = cv2.cvtColor(rightEye_img, cv2.COLOR_BGR2RGB)
        rightEye_img = cv2.flip(rightEye_img, 1)
        rightEye_img = rightEye_img / 255
        rightEye_img = rightEye_img.transpose(2, 0, 1)

        label = line[3]
        label = np.array(label).astype(float)

        return {"faceImg": torch.from_numpy(face_img).type(torch.FloatTensor),
                "leftEyeImg": torch.from_numpy(leftEye_img).type(torch.FloatTensor),
                "rightEyeImg": torch.from_numpy(rightEye_img).type(torch.FloatTensor),
                "rects": torch.from_numpy(np.array(line[2])).type(torch.FloatTensor),
                "label": torch.from_numpy(label).type(torch.FloatTensor),
                "frame": line}


def txtload(path, type, leave_out, batch_size, shuffle=False, num_workers=0):
    dataset = loader(path, type, leave_out)
    print("[Read Data]: MPIIFaceGaze Dataset")
    print("[Read Data]: Total num: {:d}".format(len(dataset)))
    print("[Read Data]: Dataset type: {:s}".format(type))
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load


