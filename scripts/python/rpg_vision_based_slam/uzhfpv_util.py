'''
Author: Wtrwater 1921852290@qq.com
Date: 2024-07-31 15:33:21
LastEditors: Wtrwater 1921852290@qq.com
LastEditTime: 2024-08-02 14:36:17
FilePath: /rpg_vision-based_slam/scripts/python/rpg_vision_based_slam/uzhfpv_util.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
# This script has been adapted from: https://github.com/uzh-rpg/uzh_fpv_open/blob/master/python/uzh_fpv/flags.py

import numpy as np
import os
import yaml

import rpg_vision_based_slam.calibration as calibration
import rpg_vision_based_slam.flags as flags
import rpg_vision_based_slam.uzhfpv_flags as uzhfpv_flags
import rpg_vision_based_slam.pose as pose


def importCamI(y, i):
    y_cam = y['cam%d' % i]
    T_C_I4 = np.array(y_cam['T_cam_imu'])
    T_C_B = pose.Pose(T_C_I4[:3, :3], T_C_I4[:3, 3:])
    timeshift_cam_imu = float(y_cam['timeshift_cam_imu'])
    dist = calibration.EquidistantDistortion(y_cam['distortion_coeffs'])
    intr = y_cam['intrinsics']
    shape = list(reversed(y_cam['resolution']))
    return calibration.CamCalibration(intr[:2], intr[2:], dist, shape, T_C_B, timeshift_cam_imu)


def readCamCalibration(cam_idx):
    calib_path = os.path.join(flags.datasetsPath(), uzhfpv_flags.calibRelativePath())
    f_calib = open(os.path.join(calib_path, 'camchain-imucam-..%s_calib_%s_imu.yaml' \
        % (uzhfpv_flags.envCamString(), uzhfpv_flags.sensorString())))
    y = yaml.load(f_calib, Loader=yaml.FullLoader)
    calibs = importCamI(y, cam_idx)
    f_calib.close()
    return calibs
