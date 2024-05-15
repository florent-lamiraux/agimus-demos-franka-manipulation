from pathlib import Path
import numpy as np
import pickle
import torch

from happypose.toolbox.inference.types import ObservationTensor
from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import CosyPoseWrapper

from config import DS_NAME, COLOR_FILE, TEST_IMG_IDS, CAM_FILE, data_dir


K = np.load(CAM_FILE)
rgb_arr = np.load(COLOR_FILE)
h,w = rgb_arr.shape[1], rgb_arr.shape[2]

if TEST_IMG_IDS == 'all':
    TEST_IMG_IDS = np.arange(len(rgb_arr))

imgs = {img_idx: rgb_arr[img_idx] for img_idx in TEST_IMG_IDS}
Ks = {img_idx: K for img_idx in TEST_IMG_IDS}


cam_indices = list(imgs.keys())
obs_tensor_lst = [ObservationTensor.from_numpy(imgs[k], None, Ks[k]) for k in cam_indices]
batched_obs = ObservationTensor(
    images=torch.cat([obs.images for obs in obs_tensor_lst]),
    K=torch.cat([obs.K for obs in obs_tensor_lst])
)

if torch.cuda.is_available():
    batched_obs.cuda()

renderer_type = 'panda3d'
# renderer_type = 'bullet'
cosypose = CosyPoseWrapper(DS_NAME, renderer_type=renderer_type, n_workers=1)
data_TCO, extra_data = cosypose.pose_predictor.run_inference_pipeline(batched_obs,
                                                                  run_detector=True,
                                                                  n_coarse_iterations=1, 
                                                                  n_refiner_iterations=4,
                                                                  detection_th=0.7,
                                                                  )

data_TCO = data_TCO.cpu()
data_TCO.infos['cam_id'] = np.array([cam_indices[batch_id] for batch_id in data_TCO.infos['batch_im_id']])

tmp_res = Path('tmp') / data_dir
tmp_res.mkdir(exist_ok=True, parents=True)
TCO_path = tmp_res / 'data_TCO.pkl'
with open(TCO_path, 'wb') as f:
    print(f'Save {TCO_path}')
    pickle.dump(data_TCO, f)
