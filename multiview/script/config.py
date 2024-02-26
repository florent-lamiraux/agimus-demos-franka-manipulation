from pathlib import Path



# Multiview m3t refinement
REFINE_MULTI = True
INIT_IMG = 0  # use nth image for cosy based initial guess (if REFINE_MULTI false)
INITIAL_GUESS_AVG = True
USE_DEPTH = False

# Multiview CosyPose refinement
USE_KNOWN_CAMERA_POSES = True

DS_NAME = 'ycbv'
OBJ_NAME = 'obj_000002'  # Cheezit
# OBJ_NAME = 'obj_000005'  # mustard


# ###############################
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



###############################
data_dir = Path('data/multiview_feb6') 
# experiment_name = 'multiview_feb6'
COLOR_FILE = data_dir / 'color_img.npy'
DEPTH_FILE = data_dir / 'depth_img.npy'
CAM_FILE = data_dir / 'camera_matrix.npy'
ROBOT_CONFIG_FILE = data_dir / 'q.npy'
CAM_POSE_FILE = data_dir / 'cam_poses.npy'

TEST_IMG_IDS = 'all'
# TEST_IMG_IDS = [0, 9, 11, 13]  # mustard
# TEST_IMG_IDS = [0 ,1 ,7 ,8 ,9 ,11 ,12 ,13]  # mustard++
# TEST_IMG_IDS = [2, 4, 5, 6, 8]  # cheezit
# TEST_IMG_IDS = [2, 4]  # cheezit
# 0


# ###############################
# data_dir = Path('data/feb5_data')
# COLOR_FILE = data_dir / 'feb5_images.npy'
# DEPTH_FILE = data_dir / 'feb5_depth.npy'
# CAM_FILE = data_dir / '../checkerboard/camera_matrix.npy.npy'
# ROBOT_CONFIG_FILE = data_dir / 'feb5_q.npy'
# CAM_POSE_FILE = data_dir / 'feb5_cam_poses.npy'

# TEST_IMG_IDS = 'all'
# TEST_IMG_IDS = [0, 9, 11, 13]  # mustard
# TEST_IMG_IDS = [0 ,1 ,7 ,8 ,9 ,11 ,12 ,13]  # mustard++
# TEST_IMG_IDS = [2, 4, 5, 6, 8]  # cheezit
# TEST_IMG_IDS = [2, 4]  # cheezit


