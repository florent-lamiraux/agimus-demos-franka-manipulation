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
import os

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

from config import CAM_FILE, TEST_IMG_IDS, CAM_POSE_FILE, USE_KNOWN_CAMERA_POSES, GIF_VIZ, REPROJ_VIZ, DS_NAME, data_dir

print(LOCAL_DATA_DIR)

import happypose.pose_estimators.cosypose.cosypose.utils.tensor_collection as tc
from happypose.toolbox.datasets.datasets_cfg import make_object_dataset


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

# To handle CUDA Exceptions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["EGL_VISIBLE_DEVICES"] = "-1"

# ligne 73
# object_ds = BOPObjectDataset(LOCAL_DATA_DIR / 'bop_datasets/tless/models_cad') #'bop_datasets/ycbv/models'
dir_path = os.getcwd()
tmp = Path(dir_path.rsplit('/script',1)[0])
# tmp = Path('tmp')
tmp_res = tmp / data_dir.name
tmp_res.mkdir(exist_ok=True)

#############################

# Enforces a 'tless-obj_000042' object label format
label_format = DS_NAME+'-{label}'
if DS_NAME == 'ycbv':
    object_ds = BOPObjectDataset(LOCAL_DATA_DIR / 'bop_datasets/ycbv/models', label_format)
elif DS_NAME == 'tless':
    object_ds = BOPObjectDataset(LOCAL_DATA_DIR / 'bop_datasets/tless/models_cad', label_format)
    # object_ds = BOPObjectDataset(LOCAL_DATA_DIR / 'bop_datasets/tless/models_cad')
mesh_db = MeshDataBase.from_object_ds(object_ds)

print('!-----[DEBUG]-----!')
print(label_format)
print(DS_NAME)
print(CAM_POSE_FILE)
print(dir_path)
print(tmp)
print(tmp_res)
print('!-----------------!')

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
with open(tmp_res / 'data_TCO.pkl', 'rb') as f:
    data_TCO = pickle.load(f)

# Candidates
# ---> infos should be === ["view_id", "group_id", "scene_id", "score", "label"] ===
data_TCO.infos = data_TCO.infos.rename(columns={'cam_id': 'view_id'})
data_TCO.infos['scene_id'] = 42
data_TCO.infos["group_id"] = 0

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
    if DS_NAME == 'tless':
        DS_NAME = 'tless.cad'
    # HACK tless-obj_000008 -> obj_000008
    # REASON:
    # - MeshDataBase for multiview RANSAC has 'obj_000008' format
    # - make_object_dataset for panda renderer creates 'tless-obj_000008' format
    labels_ds = objects_pred.infos['label']
    objects_pred.infos['label'] = labels_ds.apply(lambda label: label.split('-')[1])

    images = make_scene_renderings(objects_pred, 
                                   cameras_pred,
                                   urdf_ds_name=DS_NAME, 
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

    objects_pred.infos['label'] = labels_ds
    save_path = Path(f'gifs/{data_dir.name}.gif')
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

with open(tmp_res / 'T_bc_arr.pkl', 'wb') as f:
    pickle.dump(T_bc_arr, f)
    print('Saved T_bc_arr.pkl')
with open(tmp_res / 'T_bo_dic.pkl', 'wb') as f:
    pickle.dump(T_bo_dic, f)
    print('Saved T_bo_dic.pkl')


if REPROJ_VIZ:

    from happypose.toolbox.renderer.types import Panda3dObjectData, Panda3dCameraData, Panda3dLightData
    from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
    from happypose.toolbox.datasets.datasets_cfg import make_object_dataset

    from config import COLOR_FILE, CAM_POSE_FILE, DS_NAME

    if DS_NAME == 'tless':
        DS_NAME = 'tless.cad'

    Kc = np.load(CAM_FILE)
    rgb_arr = np.load(COLOR_FILE)
    h,w = rgb_arr.shape[1], rgb_arr.shape[2]

    TWO = objects_pred.tensors['TWO']
    TWC = cameras_pred.tensors['TWC']

    object_datas = [
        Panda3dObjectData(label=objects_pred.infos.iloc[i]['label'], TWO=TWO[i])
        for i in range(len(TWO))
    ]

    camera_datas = [
        Panda3dCameraData(
            K=Kc, resolution=(h, w), TWC=Twc
        )
        for Twc in TWC
    ]

    light_datas = [
        Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))
    ]

    asset_dataset = make_object_dataset(DS_NAME)
    renderer = Panda3dSceneRenderer(asset_dataset)

    renderings = renderer.render_scene(
        object_datas,
        camera_datas,
        light_datas,
        render_normals=True,
        render_depth=True,
        render_binary_mask=True,
    )

    overlays = []
    alpha = 0.5
    for i, view_id in enumerate(valid_view_ids):
        over = rgb_arr[view_id].copy()
        normals = renderings[i].normals
        mask = renderings[i].binary_mask
        mask = mask.repeat(3,axis=2)
        over[mask] = (1-alpha)*over[mask] + alpha * normals[mask]

        plt.figure()
        plt.imshow(over)
        fig_path = tmp_res / f'viz_multiview_rgb_{view_id}.png'
        plt.savefig(fig_path)
        print(f'Save {fig_path}')
