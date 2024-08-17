'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

import math
import os
import sys

import numpy as np
from pyquaternion import Quaternion
import yaml

import rpg_vision_based_slam.alignment as alignment
import rpg_vision_based_slam.flags as flags
import rpg_vision_based_slam.pose as pose
import rpg_vision_based_slam.utils as utils
import rpg_vision_based_slam.uzhfpv_flags as uzhfpv_flags

FLAGS = flags.FLAGS

# W: fixed world frame. Global measurements are expressed in this frame.
# G: fixed colmap frame. Spline is expressed in this frame.


def run():
    config_file = open(FLAGS.config)
    configs = yaml.load(config_file, Loader=yaml.FullLoader)
    g_meas_fn = configs['gp_fn']
    order = str(configs['spline_order'])
    spline_knot_dt = str(int(1000 * configs['spline_control_nodes_dt_s']))
    spline_fn = os.path.join(configs['colmap_spline_dir'], 'order_' + order + '/dt_' + spline_knot_dt + '_ms/spline.txt')
    res_dir = configs['spline_global_ref_alignment_dir']
    res_dir += '/order_' + order + '/dt_' + spline_knot_dt + '_ms'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Load measurements
    g_meas = np.loadtxt(g_meas_fn)
    g_ts = g_meas[:, 0]
    g_pos = g_meas[:, 1:4]

    # downsample to 20 Hz
    dt = 0.050
    if ((g_ts[1] - g_ts[0]) < (dt - 0.001)):
        t_k = g_ts[0]

        g_ts_sampled = []
        g_pos_sampled = []
        g_ts_sampled.append(t_k)
        g_pos_sampled.append(g_pos[0])

        for i, t in enumerate(g_ts):
            if t >= (dt + t_k):
                t_k = t
                g_ts_sampled.append(t_k)
                g_pos_sampled.append(g_pos[i])

        g_ts = np.asarray(g_ts_sampled)
        g_pos = np.asarray(g_pos_sampled)

    spline = np.loadtxt(spline_fn)
    spline_ts = spline[:, 0]
    spline_pos = spline[:, 1:4]

    # Plots
    if (FLAGS.gui):
        plt.figure(0)
        plots.xyPlot('Before initial alignment', g_pos[:, 0:2], 'global measurements', spline_pos[:, 0:2], 'spline')
        plt.show()

    # Initialize alignment: T_wg and scale.
    # time snap = time leica + init_t_offset
    init_t_offset = configs['init_t_offset_cam_gp']
    print('Using initial time offset: %.3f' % init_t_offset)
    g_ts_aligned = np.asarray([t - init_t_offset for t in g_ts])

    import ipdb
    ipdb.set_trace()

    idx_t_matches = alignment.associateTimestamps(spline_ts, g_ts_aligned)
    idx_t_spline = [idx[0] for idx in idx_t_matches]
    idx_t_g = [idx[1] for idx in idx_t_matches]

    spline_pos_to_align = spline_pos[idx_t_spline, :]
    g_pos_to_align = g_pos[idx_t_g, :]

    assert spline_pos_to_align.shape == g_pos_to_align.shape

    scale, R_wg, p_wg = alignment.alignUmeyama(g_pos_to_align, spline_pos_to_align)
    T_wg = pose.Pose(R_wg, p_wg.reshape(3, 1))

    # save results
    np.savetxt(os.path.join(res_dir, 'T_wg_init.txt'), T_wg.asArray())
    np.savetxt(os.path.join(res_dir, 'scale_init.txt'), np.array([scale]))

    print('saving to: %s' % res_dir)
    print('T_wg:')
    print(T_wg.asArray())
    print('scale: %.3f' % scale)

    if (FLAGS.gui):
        spline_pos_aligned = np.asarray([scale * np.dot(T_wg.R, p) + T_wg.t.ravel() for p in spline_pos])
        plt.figure(1)
        plots.xyPlot('After initial alignment', g_pos[:, 0:2], 'global measurements', spline_pos_aligned[:, 0:2], 'spline')
        plt.show()


if __name__ == '__main__':
    sys.argv = flags.FLAGS(sys.argv)
    if (FLAGS.gui):
        import rpg_vision_based_slam.plots as plots
        import matplotlib.pyplot as plt
    run()
