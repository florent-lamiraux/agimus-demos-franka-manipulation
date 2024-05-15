import cv2
from pathlib import Path
import numpy as np
import pickle
import pinocchio as pin

from olt.tracker import Tracker
from olt.config import OBJ_MODEL_DIRS, TrackerConfig, CameraConfig, RegionModalityConfig, DepthModalityConfig
from olt.utils import Kres2intrinsics


from config import (DS_NAME, COLOR_FILE, DEPTH_FILE, TEST_IMG_IDS, CAM_FILE, 
                    CAM_POSE_FILE, OBJ_NAME, REFINE_MULTI, INITIAL_GUESS_AVG, INIT_IMG, USE_DEPTH)

tmp = Path('tmp')
tmp.mkdir(exist_ok=True)

#################
# Reading dataset
#################

Kc = np.load(CAM_FILE)
T_wc_arr = np.load(CAM_POSE_FILE)
rgb_arr = np.load(COLOR_FILE)
depth_arr = np.load(DEPTH_FILE)

if TEST_IMG_IDS == 'all':
    TEST_IMG_IDS = np.arange(len(rgb_arr))

hc,wc = rgb_arr.shape[1], rgb_arr.shape[2]
intr_color = Kres2intrinsics(Kc,wc,hc)

# Depth cam parameters
fx_d, fy_d = 383.903, 383.903
cx_d, cy_d = 320.794, 228.177  
hd,wd = depth_arr.shape[1], depth_arr.shape[2]

intr_depth = {
        'fu': fx_d, 
        'fv': fy_d, 
        'ppu': cx_d, 
        'ppv': cy_d,
        'width': wd,
        'height': hd, 
    }

R_dc = np.array([0.999939, -0.00885668, 0.00662281, 
                 0.00887612, 0.999956, -0.00291152, 
                 -0.00659674, 0.00297013, 0.999974]).reshape((3,3))
R_dc = R_dc.T
t_dc = np.array([-0.0147111, -4.32366e-05, -0.000319071])
M_dc = pin.SE3(R_dc, t_dc)
M_cd = M_dc.inverse()

#################
# Creating tracker + localizer
#################

def initial_pose_1frame_cosy(T_wc_arr, img_ids, T_co_preds, init_img_id):
    T_wc_init = T_wc_arr[img_ids[init_img_id]]
    T_co_init_dic = T_co_preds[img_ids[init_img_id]]
    T_wo_init = {k: T_wc_init @ T_co for k, T_co in T_co_init_dic.items()}
    return T_wo_init


def initial_pose_multi_cosy(T_wc_arr, img_ids, T_co_preds, obj_name):
    T_wc_lst = [T_wc_arr[i] for i in img_ids]
    # TODO: check detection in all frames -> fail if not detect
    T_co_lst = [T_co_dic[obj_name] for T_co_dic in T_co_preds.values()] 

    T_wo_lst = [T_wc@T_co for T_wc, T_co in zip(T_wc_lst, T_co_lst)]

    # TODO: avg for all objects
    return {obj_name: se3_avg(T_wo_lst)}

def se3_avg(T_lst):
    R_arr = np.array([T[:3,:3] for T in T_lst])
    t_arr = np.array([T[:3,3] for T in T_lst])
    R_avg = R_arr.mean(axis=0)
    t_avg = t_arr.mean(axis=0)

    # projection on SO(3)
    u, s, vh = np.linalg.svd(R_avg)
    R_proj = u @ vh

    T_avg = np.eye(4)
    T_avg[:3,:3] = R_proj
    T_avg[:3,3] = t_avg

    return T_avg

imgs_rgb = {img_idx: rgb_arr[img_idx] for img_idx in TEST_IMG_IDS}
imgs_depth = {img_idx: depth_arr[img_idx] for img_idx in TEST_IMG_IDS}
Ks = {img_idx: Kc for img_idx in TEST_IMG_IDS}

# Load cosy batch single-view estimates
with open(tmp / 'T_co_preds.pkl', 'rb') as f:
    print('Load T_co_preds.pkl')
    T_co_preds = pickle.load(f)

# Initial guess from one of the views and CosyPose batch prediction 
if INITIAL_GUESS_AVG:
    T_wo_init = initial_pose_multi_cosy(T_wc_arr, TEST_IMG_IDS, T_co_preds, OBJ_NAME)
    print('initial_pose_multi_cosy')
    print(T_wo_init)
else:
    T_wo_init = initial_pose_1frame_cosy(T_wc_arr, TEST_IMG_IDS, T_co_preds, INIT_IMG)
    print('initial_pose_1frame_cosy')
    print(T_wo_init)


# Refine pose using multiple views
camera_cfgs = {
    img_idx: CameraConfig(
        color_intrinsics=intr_color,
        color2world_pose=T_wc_arr[img_idx],
        depth_intrinsics=intr_depth,
        depth2world_pose=T_wc_arr[img_idx] @ M_cd.homogeneous,
    )
    for img_idx in TEST_IMG_IDS
}

# PARAMS tracker
n_corr_iterations = 6
n_update_iterations = 4
scales = [24, 12, 6, 4, 2, 1]
standard_deviations = [20., 20., 15.0, 5.0, 3.5, 1.5]
#############

region_modalities = {
    img_idx: RegionModalityConfig(
        # scales=scales,
        # standard_deviations=standard_deviations,
        # visualize_lines_correspondence=False
    )
    for img_idx in TEST_IMG_IDS
}
depth_modalities = {
    img_idx: DepthModalityConfig()
    for img_idx in TEST_IMG_IDS
}

accepted_objs = [OBJ_NAME]
tracker_cfg = TrackerConfig(
    cameras=camera_cfgs,
    region_modalities=region_modalities,
    depth_modalities=depth_modalities if USE_DEPTH else {}
) 
tracker_cfg.viewer_display = True
tracker_cfg.viewer_save = True
tracker_cfg.n_corr_iterations = n_corr_iterations
tracker_cfg.n_update_iterations = n_update_iterations

tracker = Tracker(OBJ_MODEL_DIRS[DS_NAME], accepted_objs, tracker_cfg)
tracker.init()
# tracker.tracker.n_corr_iterations = 100
# tracker.tracker.n_update_iterations = 100

# RGB -> BGR for, only for proper opencv viz (does not change estimation)
imgs_rgb = {img_idx: cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img_idx, img in imgs_rgb.items()}

tracker.set_images(imgs_rgb, imgs_depth)
tracker.detected_bodies(T_wo_init)
T_wo_dic, scores = tracker.get_current_preds()
tracker.update_viewers()
cv2.waitKey(0)
print('DONE')
if REFINE_MULTI:
    print('TRACK 1')
    dt = tracker.track()
    print('TRACK 2')
    dt = tracker.track()
    print('TRACK 3')
    dt = tracker.track()
    dt = tracker.track()

with open(tmp / 'T_wo_dic.pkl', 'wb') as f:
    print('Save T_wo_dic.pkl')
    pickle.dump(T_wo_dic, f)

# print('track took (ms):', 1000*dt)
tracker.update_viewers()
cv2.waitKey(0)
print('DONE')

T_wo_dic, scores = tracker.get_current_preds()
