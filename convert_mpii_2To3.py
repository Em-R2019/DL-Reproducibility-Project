import os
import numpy as np
import yaml
import data_processing_core as dpc
import gazeconversion as co
import cv2
from scipy import io as sio

scale = False

def read_file(path, subject):
    with open(path) as infile:
        lines = infile.readlines()
        lines.pop(0)

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

    # Convert pixel to mm
    screen = sio.loadmat(screen_size)
    w_pixel = screen["width_pixel"][0][0]
    h_pixel = screen["height_pixel"][0][0]
    w_mm = screen["width_mm"][0][0]
    h_mm = screen["height_mm"][0][0]
    w_ratio = w_mm / w_pixel
    h_ratio = h_mm / h_pixel
    ratio = np.array([w_ratio, h_ratio])

    # read gaze origin, rmat and smat from annotation file
    annotation = os.path.join(annofolder, f"{person}.label")
    with open(annotation) as infile:
        lines = infile.readlines()
        lines.pop(0)

    total_loss = 0
    count = 0
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

        if scale == True :
            pred_norm = np.dot(Smat, np.dot(Rmat, np.reshape(pred_ccs, (3, 1))))
            gt_norm = np.dot(Smat, np.dot(Rmat, np.reshape(gt_ccs, (3, 1))))
        else:
            pred_norm = np.dot(Rmat, np.reshape(pred_ccs, (3, 1)))
            gt_norm = np.dot(Rmat, np.reshape(gt_ccs, (3, 1)))

        pred_norm = pred_norm.flatten() / np.linalg.norm(pred_norm)
        gt_norm = gt_norm.flatten() / np.linalg.norm(gt_norm)
        gaze_loss = dpc.AngularLoss(pred_norm, gt_norm)
        total_loss += gaze_loss

        if isprint:
            print(name, pred_norm, gt_norm, gaze_loss)

        count = count + 1

    return total_loss, count


def main(name, logfolder, calibrationfolder, labelfolder):
    total_loss = 0
    total_count = 0

    for person in ["p00", "p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09", "p10", "p11", "p12", "p13",
                   "p14"]:
        logfile = os.path.join(logfolder, f"{name}.log")
        screenPose = os.path.join(calibrationfolder, f"{person}/Calibration/monitorPose.mat")
        screenSize = os.path.join(calibrationfolder, f"{person}/Calibration/screenSize.mat")

        gaze_loss, count = convert_mpii_2T3(screenPose,
                                            screenSize,
                                            logfile,
                                            labelfolder,
                                            person
                                            )

        total_loss += gaze_loss
        total_count += count
    return total_loss / total_count, total_count


if __name__ == "__main__":
    evaluation_path = "models/model10/evaluation/"
    calibration_path = "data/MPIIFaceGaze/"
    label_path = "data/output2/Label/"

    epoch_log_3D = open(os.path.join(evaluation_path, "epoch3D-False.log"), 'w')

    config = yaml.safe_load(open("configs/configp08.yaml"))
    config = config["test"]["load"]
    tests = range(config["begin_step"], config["end_step"] + 1)

    for test in tests:
        acc, count = main(str(test), evaluation_path, calibration_path, label_path)

        loger = f"[{test}] Total Num: {count}, acc: {acc} \n"
        print(loger)
        epoch_log_3D.write(loger)
