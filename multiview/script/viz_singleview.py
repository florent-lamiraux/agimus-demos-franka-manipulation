import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from happypose.toolbox.renderer.types import Panda3dObjectData, Panda3dCameraData, Panda3dLightData
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.datasets.datasets_cfg import make_object_dataset

from config import COLOR_FILE, CAM_POSE_FILE, DS_NAME, data_dir, CAM_FILE


tmp = Path('tmp')
tmp_res = tmp / data_dir
# tmp_res = tmp / data_dir.name
tmp_res.mkdir(exist_ok=True, parents=True)

# Load results from cosypose singleview batch estimations
with open(tmp_res / 'data_TCO.pkl', 'rb') as f:
    data_TCO = pickle.load(f)

if DS_NAME == 'tless':
    DS_NAME = 'tless.cad'

asset_dataset = make_object_dataset(DS_NAME)
renderer = Panda3dSceneRenderer(asset_dataset)

Kc = np.load(CAM_FILE)
rgb_arr = np.load(COLOR_FILE)
h,w = rgb_arr.shape[1], rgb_arr.shape[2]

camera_datas = [
    Panda3dCameraData(
        K=Kc, resolution=(h, w), TWC=np.eye(4)
    )
]

light_datas = [
    Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))
]

for i, rgb in enumerate(rgb_arr):

    img_infos = data_TCO.infos[data_TCO.infos['batch_im_id'] == i]
    labels = img_infos['label']
    poses = data_TCO.tensors['poses'][img_infos.index,:,:]

    object_datas = [
        Panda3dObjectData(label=label, TWO=T_bo)
        for label, T_bo in zip(labels, poses)
    ]

    renderings = renderer.render_scene(
        object_datas,
        camera_datas,
        light_datas,
        render_normals=True,
        render_depth=True,
        render_binary_mask=True,
    )

    alpha = 0.5
    over = rgb.copy()
    normals = renderings[0].normals
    mask = renderings[0].binary_mask
    mask = mask.repeat(3,axis=2)
    over[mask] = (1-alpha)*over[mask] + alpha * normals[mask]

    plt.figure()
    plt.imshow(over)
    fig_path = tmp_res / f'viz_singleview_rgb_{i}.png'
    plt.savefig(fig_path)
    print(f'Save {fig_path}')
