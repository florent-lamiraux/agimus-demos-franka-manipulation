import os
from pathlib import Path



# Multiview m3t refinement
REFINE_MULTI = True
INIT_IMG = 0  # use nth image for cosy based initial guess (if REFINE_MULTI false)
INITIAL_GUESS_AVG = False
USE_DEPTH = True

# Multiview CosyPose refinement
USE_KNOWN_CAMERA_POSES = False
GIF_VIZ = False
REPROJ_VIZ = True

# OBJ_NAME = 'obj_000002'  # Cheezit
# OBJ_NAME = 'obj_000005'  # mustard


# ###############################
# DS_NAME = 'ycbv'
# data_dir = Path('data/feb1_mustard')
# # COLOR_FILE = data_dir /'img_np_list_21.npy'
# COLOR_FILE = data_dir / 'img_np_list_21_standing.npy'
# DEPTH_FILE = None
# CAM_FILE = data_dir / 'camera_matrix.npy'
# ROBOT_CONFIG_FILE = data_dir / 'q_list_21_np.npy'
# CAM_POSE_FILE = data_dir / 'robot_base_to_cam_poses_21.npy'

# # TEST_IMG_IDS = list(range(21))
# # TEST_IMG_IDS = [0, 10, 12, 15, 18, 20]  # lying
# TEST_IMG_IDS = [0,2,4,6,7,9]  # standing



# ###############################
# DS_NAME = 'ycbv'
# data_dir = Path('data/multiview_feb6') 
# # experiment_name = 'multiview_feb6'
# COLOR_FILE = data_dir / 'color_img.npy'
# DEPTH_FILE = data_dir / 'depth_img.npy'
# CAM_FILE = data_dir / 'camera_matrix.npy'
# ROBOT_CONFIG_FILE = data_dir / 'q.npy'
# CAM_POSE_FILE = data_dir / 'cam_poses.npy'

# TEST_IMG_IDS = 'all'
# # TEST_IMG_IDS = [0, 9, 11, 13]  # mustard
# # TEST_IMG_IDS = [0 ,1 ,7 ,8 ,9 ,11 ,12 ,13]  # mustard++
# # TEST_IMG_IDS = [2, 4, 5, 6, 8]  # cheezit
# # TEST_IMG_IDS = [2, 4]  # cheezit
# # 0


# ###############################
# DS_NAME = 'ycbv'
# data_dir = Path('data/feb5_data')
# COLOR_FILE = data_dir / 'feb5_images.npy'
# DEPTH_FILE = data_dir / 'feb5_depth.npy'
# CAM_FILE = data_dir / 'camera_matrix.npy'
# ROBOT_CONFIG_FILE = data_dir / 'feb5_q.npy'
# CAM_POSE_FILE = data_dir / 'feb5_cam_poses.npy'

# TEST_IMG_IDS = 'all'
# # TEST_IMG_IDS = [2]


# ###############################
# DS_NAME = 'tless'
# data_dir = Path('data/multiview_tless')
# COLOR_FILE = data_dir / 'color_img.npy'
# DEPTH_FILE = data_dir / 'depth_img.npy'
# CAM_FILE = data_dir / 'camera_matrix.npy'
# ROBOT_CONFIG_FILE = data_dir / 'feb5_q.npy'
# CAM_POSE_FILE = data_dir / 'cam_poses.npy'

# TEST_IMG_IDS = 'all'
# # TEST_IMG_IDS = [2]


###############################
# DS_NAME = 'tless'
# DS_NAME = 'ycbv'
DS_NAME = 'tless'
dir = os.getcwd()
data_dir = Path((str(dir).rsplit('/script',1)[0] + '/multiview_tless_2'))
COLOR_FILE = data_dir / 'color_img.npy'
# DEPTH_FILE = data_dir / 'depth_img.npy'
CAM_FILE = data_dir / 'camera_matrix.npy'
ROBOT_CONFIG_FILE = data_dir / 'q.npy'
CAM_POSE_FILE = data_dir / 'cam_poses.npy'

TEST_IMG_IDS = 'all'
# TEST_IMG_IDS = [2]

# dir = os.getcwd()
# data_dir = Path((str(dir).rsplit('/script',1)[0] + '/multiview_tless_2'))
# COLOR_FILE = data_dir / 'color_img.npy'
# DEPTH_FILE = data_dir / 'depth_img.npy'
# CAM_FILE = data_dir / 'camera_matrix.npy'
# ROBOT_CONFIG_FILE = data_dir / 'q.npy'
# CAM_POSE_FILE = data_dir / 'cam_poses.npy'
