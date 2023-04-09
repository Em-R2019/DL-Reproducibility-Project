import os
import sys
import numpy as np
import yaml
import data_processing_core as dpc
import gazeconversion as co
import cv2
from scipy import io as sio
import argparse


def main(args):
    total_loss = 0
    total_count = 0

    # for i in ["p00", "p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09", "p10", "p11", "p12", "p13", "p14"]:
    for i in ["p11", "p12", "p13", "p14"]:
        logfile = os.path.join(args.logfolder, f"{args.name}")

        gaze_loss, count = convert_mpii_2T3(args.screenPose,
                                            args.screenSize,
                                            logfile,
                                            args.label,
                                            i
                                            # args.ispixel,
                                            # args.isprint,
                                            # args.expand
                                            )

        total_loss += gaze_loss
        total_count += count
    return total_loss / total_count, total_count


def read_file(path, subject):
    with open(path) as infile:
        lines = infile.readlines()
        lines.pop(0)
        # acc = lines.pop(-1)

    for line in lines:
        line = line.strip().split(",")

        gt = line[4:6]
        name = line[0]
        prediction = line[2:4]

        gt = np.array(gt, dtype="float")
        prediction = np.array(prediction, dtype='float')
        if name == subject:
            yield name, gt, prediction
        else:
            continue


def convert_mpii_2T3(screen_pose, screen_size, logfile, annofolder, person, ispixel=True, isprint=False, expand=1):
    # Read Screen-Camera pose
    screen_pose = sio.loadmat(screen_pose)
    rvec = screen_pose["rvects"]
    tvec = screen_pose["tvecs"]
    rmat = cv2.Rodrigues(rvec)[0]

    # Convet pixel to mm
    screen = sio.loadmat(screen_size)
    w_pixel = screen["width_pixel"][0][0]
    h_pixel = screen["height_pixel"][0][0]
    w_mm = screen["width_mm"][0][0]
    h_mm = screen["height_mm"][0][0]
    w_ratio = w_mm / w_pixel
    h_ratio = h_mm / h_pixel
    ratio = np.array([w_ratio, h_ratio])

    # read gaze origin from annotation file
    # person = os.path.split(logfile)[0]
    # person = os.path.split(person)[1].split(".")[0]
    # person = "p00"
    annotation = os.path.join(annofolder, f"{person}.label")
    with open(annotation) as infile:
        lines = infile.readlines()
        lines.pop(0)

    total_loss = 0
    for count, (name, gt, prediction) in enumerate(read_file(logfile, person)):
        annos = lines[count].strip().split(" ")

        if ispixel:
            prediction = prediction * ratio * expand
            gt = gt * ratio * expand
        else:
            prediction = prediction * expand
            gt = gt * expand

        Rvec = np.array(list(map(eval, annos[-3].split(","))))
        Svec = np.array(list(map(eval, annos[-2].split(","))))
        origin = np.array(list(map(eval, annos[-1].split(","))))

        Rmat = cv2.Rodrigues(Rvec)[0]
        Smat = np.diag(Svec)
        mat = np.dot(np.linalg.inv(Rmat), np.linalg.inv(Smat))

        origin_ccs = np.dot(mat, np.reshape(origin, (3, 1)))

        pred_ccs = co.Gaze2DTo3D(prediction, origin_ccs.flatten(), rmat, tvec)
        gt_ccs = co.Gaze2DTo3D(gt, origin_ccs.flatten(), rmat, tvec)

        pred_norm = np.dot(Smat, np.dot(Rmat, np.reshape(pred_ccs, (3, 1))))
        gt_norm = np.dot(Smat, np.dot(Rmat, np.reshape(gt_ccs, (3, 1))))
        pred_norm = pred_norm.flatten() / np.linalg.norm(pred_norm)
        gt_norm = gt_norm.flatten() / np.linalg.norm(gt_norm)
        gaze_loss = dpc.AngularLoss(pred_norm, gt_norm)
        total_loss += gaze_loss

        if isprint:
            print(name, pred_norm, gt_norm, gaze_loss)

    return total_loss, count + 1


if __name__ == "__main__":
    evaluation_path = "model 05042023/evaluation/"

    parser = argparse.ArgumentParser(description="Convert 3D gaze to 2D gaze on MPIIGaze")

    parser.add_argument('--screenPose', type=str, default="data/MPIIFaceGaze/p00/Calibration/monitorPose.mat")

    parser.add_argument('--screenSize', type=str, default="data/MPIIFaceGaze/p00/Calibration/screenSize.mat")

    parser.add_argument('--logfolder', type=str, default=evaluation_path)

    parser.add_argument('--label', type=str, default="data/output2/Label/")

    parser.add_argument('--name', type=str, default="log")

    args = parser.parse_args()

    epoch_log_3D = open(os.path.join(evaluation_path, "epoch3D.log"), 'w')

    config = yaml.safe_load(open("config.yaml"))
    config = config["test"]["load"]
    tests = range(config["begin_step"], config["end_step"] + 1)

    for test in tests:
        args.name = str(test) + ".log"
        acc, count = main(args)
        loger = f"[{test}] Total Num: {count}, acc: {acc} \n"
        print(loger)
        epoch_log_3D.write(loger)