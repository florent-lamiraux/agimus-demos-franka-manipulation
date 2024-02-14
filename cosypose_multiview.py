
import argparse
import numpy as np
import torch
import time
import pickle
from pathlib import Path
import pandas as pd
from collections import defaultdict
import imageio
import matplotlib.pyplot as plt

import happypose.pose_estimators.cosypose.cosypose.utils.tensor_collection as tc

from happypose.pose_estimators.cosypose.cosypose.config import (
    LOCAL_DATA_DIR,
)
from happypose.pose_estimators.cosypose.cosypose.datasets.bop_object_datasets import (
    BOPObjectDataset,
)
from happypose.pose_estimators.cosypose.cosypose.integrated.multiview_predictor import (
    MultiviewScenePredictor,
)
from happypose.pose_estimators.cosypose.cosypose.lib3d.rigid_mesh_database import (
    MeshDataBase,
)
from happypose.pose_estimators.cosypose.cosypose.visualization.multiview import make_scene_renderings

from config import CAM_FILE, TEST_IMG_IDS, CAM_POSE_FILE, USE_KNOWN_CAMERA_POSES

from agimus_demos.calibration import HandEyeCalibration as Calibration


GIF_VIZ = False

parser = argparse.ArgumentParser(
    "CosyPose multi-view reconstruction for a custom scenario",
)
parser.add_argument(
    "--sv_score_th",
    default=0.3,
    type=int,
    help="Score to filter single-view predictions",
)
parser.add_argument(
    "--ransac_n_iter",
    default=2000,
    type=int,
    help="Max number of RANSAC iterations per pair of views",
)
parser.add_argument(
    "--ransac_dist_threshold",
    default=0.02,
    type=float,
    help="Threshold (in meters) on symmetric distance to consider "
    "a tentative match an inlier",
)
parser.add_argument(
    "--ba_n_iter",
    default=10,
    type=int,
    help="Maximum number of LM iterations in stage 3",
)
args = parser.parse_args()


tmp = Path('tmp')
tmp.mkdir(exist_ok=True)

#############################


object_ds = BOPObjectDataset(LOCAL_DATA_DIR / 'bop_datasets/ycbv/models')
mesh_db = MeshDataBase.from_object_ds(object_ds)
mv_predictor = MultiviewScenePredictor(mesh_db)

TBC_fk = torch.from_numpy(np.load(CAM_POSE_FILE))
Nc = len(TBC_fk)
if TEST_IMG_IDS == 'all':
    TEST_IMG_IDS = np.arange(Nc)

K = np.load(CAM_FILE)
K = torch.as_tensor(np.stack(Nc*[K]))
cameras = tc.PandasTensorCollection(K=K, 
                                    TWC=TBC_fk.float(),
                                    infos=pd.DataFrame({"view_id": TEST_IMG_IDS}))
cameras.infos["scene_id"] = 1
cameras.infos["batch_im_id"] = 0

# Load results from cosypose singleview batch estimations
with open(tmp / 'data_TCO.pkl', 'rb') as f:
    data_TCO = pickle.load(f)

# Candidates
# ---> infos should be === ["view_id", "group_id", "scene_id", "score", "label"] ===
data_TCO.infos = data_TCO.infos.rename(columns={'cam_id': 'view_id'})
data_TCO.infos['scene_id'] = 42
data_TCO.infos["group_id"] = 0
# ycbv-obj_000008 -> obj_000008
data_TCO.infos['label'] = data_TCO.infos['label'].apply(lambda label: label.split('-')[1])

t1 = time.time()
predictions = mv_predictor.predict_scene_state(
    candidates=data_TCO,
    cameras=cameras,
    score_th=args.sv_score_th,
    use_known_camera_poses=USE_KNOWN_CAMERA_POSES,
    ransac_n_iter=args.ransac_n_iter,
    ransac_dist_threshold=args.ransac_dist_threshold,
    ba_n_iter=args.ba_n_iter,
)

print('CosyMulti took (ms): ', 1000*(time.time() - t1))

objects_pred = predictions['scene/objects']
cameras_pred = predictions['scene/cameras']

if GIF_VIZ:
    print('Generating scene GIF...')
    fps = 10
    duration = 20
    n_images = fps * duration
    # n_images = 1  # Uncomment this if you just want to look at one image, generating the gif takes some time
    images = make_scene_renderings(objects_pred, 
                                   cameras_pred,
                                   urdf_ds_name='ycbv', 
                                   distance=1.5, 
                                   object_scale=1.0,
                                   camera_scale=1.0,
                                   show_cameras=True, 
                                   camera_color=(0, 0, 1.0, 0.3),
                                   theta=np.pi/4, 
                                   resolution=(640, 480),
                                   object_id_ref=0, 
                                   colormap_rgb=defaultdict(lambda: [1, 1, 1, 1]),
                                   angles=np.linspace(0, 2*np.pi, n_images)
                                   )

    save_path = Path('gifs/test.gif')
    save_path.parent.mkdir(exist_ok=True)
    print(f"Save GIF from {len(images)} images at {Path(save_path).resolve()}")
    imageio.mimsave(save_path, images, fps=fps)

###########################
# Align camera poses wrt FK
###########################
    
# Retrieve CAMERA multicosy predictions
T_wc_arr = cameras_pred.tensors['TWC'].numpy()

# Select FK CAMERA poses with valid cosypose multiview views
valid_view_ids = cameras_pred.infos['view_id']
T_bc_arr = TBC_fk[valid_view_ids.to_list()].numpy()

from frames_alignment import align_robot_fk_cosy_multi
T_bw_align = align_robot_fk_cosy_multi(T_wc_arr, T_bc_arr)

# Retrieve object preds as dic
T_wo_dic = {
    objects_pred.infos['label'][i]: objects_pred.tensors['TWO'][i].numpy()  
    for i in range(len(objects_pred.infos))
}

# Transform in robot frame
T_bo_dic = {
    label: T_bw_align.homogeneous @ T_wo  
    for label, T_wo in T_wo_dic.items()
}

with open(tmp / 'T_bc_arr.pkl', 'wb') as f:
    pickle.dump(T_bc_arr, f)
    print('Saved T_bc_arr.pkl')
with open(tmp / 'T_bo_dic.pkl', 'wb') as f:
    pickle.dump(T_bo_dic, f)
    print('Saved T_bo_dic.pkl')


# Use Tracker to visualize results in img frame: 
# DIRTY: should use renderer_vispy instead 
REPROJ_VIZ = True
if REPROJ_VIZ:

    import cv2
    from olt.tracker import Tracker
    from olt.config import OBJ_MODEL_DIRS, TrackerConfig, CameraConfig
    from olt.utils import Kres2intrinsics


    def tracker_based_image_viz(Kc, rgb_arr, depth_arr, T_c_arr, T_o_dic, valid_view_ids, ds_name):
        hc,wc = rgb_arr.shape[1], rgb_arr.shape[2]
        intr_color = Kres2intrinsics(Kc,wc,hc)

        camera_cfgs = {
            view_idx: CameraConfig(
                color_intrinsics=intr_color,
                color2world_pose=T_c_arr[pose_idx]
            )
            for pose_idx, view_idx in enumerate(valid_view_ids)
        }
        accepted_objs = 'all'
        tracker_cfg = TrackerConfig(
            cameras=camera_cfgs,
        ) 
        tracker_cfg.viewer_display = True
        tracker_cfg.viewer_save = False

        tracker = Tracker(OBJ_MODEL_DIRS[ds_name], accepted_objs, tracker_cfg)
        tracker.init()

        imgs = {img_idx: rgb_arr[img_idx] for img_idx in valid_view_ids}
        imgs = {img_idx: cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img_idx, img in imgs.items()}

        tracker.set_images(imgs)
        tracker.detected_bodies(T_o_dic)
        tracker.update_viewers()


    from config import COLOR_FILE, DEPTH_FILE, TEST_IMG_IDS, CAM_POSE_FILE, DS_NAME

    Kc = np.load(CAM_FILE)
    rgb_arr = np.load(COLOR_FILE)
    depth_arr = np.load(DEPTH_FILE)

    # tracker_based_image_viz(Kc, rgb_arr, depth_arr, T_wc_arr, T_wo_dic, valid_view_ids, DS_NAME)
    tracker_based_image_viz(Kc, rgb_arr, depth_arr, T_bc_arr, T_bo_dic, valid_view_ids, DS_NAME)

def poses_multi_view():
    connectedToRos = True
    if connectedToRos:
        ri = RosInterface(robot)
        q = ri.getCurrentConfig(q0)
    else:
        q = q0[:]

    q_init = ri.getCurrentConfig(q0)
    res, q_init, err = graph.applyNodeConstraints('free', q_init)
    path = Calibration.generateConfigurationsAndPaths(q_init, 10)

if __name__ == '__main__':
    print("Start Cosypose multiview test !")
    from hpp.corbaserver.manipulation import Robot, ConstraintGraph, Constraints
    from tools_hpp import RosInterface
    from numpy import pi

    q0 = [0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4, 0.035, 0.035,
      0, 0, 1.2, 0, 0, 0, 1,
      0, 0, 0.761, 0, 0, 0, 1]
    robot = Robot("robot", "pandas", rootJointType="anchor")
    graph = ConstraintGraph(robot, 'graph')
    graph.addConstraints(graph=True,
                     constraints = Constraints(numConstraints =
                        ['locked_finger_1', 'locked_finger_2']))
    graph.initialize()

    pre_estimation_poses = []

