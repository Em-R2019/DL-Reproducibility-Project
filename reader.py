import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch
import json
import random
import copy

from tqdm import tqdm


# randomly move the bounding box around
def aug_line(line, width, height):
    bbox = np.array(line[2:5])
    bias = round(30 * random.uniform(-1, 1))
    bias = max(np.max(-bbox[0, [0, 2]]), bias)
    bias = max(np.max(-2 * bbox[1:, [0, 2]] + 0.5), bias)

    line[2][0] += int(round(bias))
    line[2][1] += int(round(bias))
    line[2][2] += int(round(bias))
    line[2][3] += int(round(bias))

    line[3][0] += int(round(0.5 * bias))
    line[3][1] += int(round(0.5 * bias))
    line[3][2] += int(round(0.5 * bias))
    line[3][3] += int(round(0.5 * bias))

    line[4][0] += int(round(0.5 * bias))
    line[4][1] += int(round(0.5 * bias))
    line[4][2] += int(round(0.5 * bias))
    line[4][3] += int(round(0.5 * bias))

    line[5][2] = line[2][2] / width
    line[5][3] = line[2][0] / height

    line[5][6] = line[3][2] / width
    line[5][7] = line[3][0] / height

    line[5][10] = line[4][2] / width
    line[5][11] = line[4][0] / height
    return line


class loader(Dataset):

    def __init__(self, data_path, data_type):
        self.lines = []
        self.labels = {}
        self.data_path = data_path
        self.data_type = data_type
        label_path = data_path + "\\Label"
        subjects = os.listdir(label_path)
        subjects.sort()
        for subject in tqdm(subjects):

            with open(os.path.join(label_path, subject), "r") as file:
                for line in file:
                    labels = line.strip().split(' ')
                    if labels[0] == "Face":
                        continue

                    labels = [i.split(',') for i in labels]
                    # labels = np.array(labels)
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
                    # label = np.array(label).astype('float')

                    rects = [rightEyeCorner[0], rightEyeCorner[1], rightEyeWidth, rightEyeHeight, faceCorner[0],
                             faceCorner[1], faceWidth, faceHeight, leftEyeCorner[0], leftEyeCorner[1], leftEyeWidth,
                             leftEyeHeight]
                    temp = [labels[0][0].split("\\")[0], labels[0][0], faceCorner, leftEyeCorner, rightEyeCorner, rects,
                            label, labels[1][0], labels[2][0]]
                    self.lines.append(temp)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        #   0        1     2     3     4     5     6
        # subject, name, face, left, right, rect, 8pts
        #
        line = self.lines[idx]

        Image_path = os.path.join(self.data_path, 'Image')

        face_img = cv2.imread(os.path.join(Image_path, line[1]))
        leftEye_img = cv2.imread(os.path.join(Image_path, line[7]))
        rightEye_img = cv2.imread(os.path.join(Image_path, line[8]))

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

        label = line[6]
        label = np.array(label).astype(float)

        return {"faceImg": torch.from_numpy(face_img).type(torch.FloatTensor),
                "leftEyeImg": torch.from_numpy(leftEye_img).type(torch.FloatTensor),
                "rightEyeImg": torch.from_numpy(rightEye_img).type(torch.FloatTensor),
                "rects": torch.from_numpy(np.array(line[5])).type(torch.FloatTensor),
                "label": torch.from_numpy(label).type(torch.FloatTensor),
                "frame": line}


def txtload(path, type, batch_size, shuffle=False, num_workers=0):
    dataset = loader(path, type)
    print("[Read Data]: MPIIFaceGaze Dataset")
    print("[Read Data]: Total num: {:d}".format(len(dataset)))
    print("[Read Data]: Dataset type: {:s}".format(type))
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load


