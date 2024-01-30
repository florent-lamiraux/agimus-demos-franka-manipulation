# Copyright 2022 CNRS
# Author: Florent Lamiraux
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from math import pi, sqrt
from hpp.corbaserver import loadServerPlugin, shrinkJointRange
from hpp.corbaserver.manipulation import Robot, \
    createContext, newProblem, ProblemSolver, ConstraintGraph, \
    ConstraintGraphFactory, CorbaClient, SecurityMargins
from hpp.gepetto.manipulation import ViewerFactory
from hpp.corbaserver import wrap_delete
from tools_hpp import displayGripper, displayHandle, generateTargetConfig, \
    shootPartInBox, RosInterface
from bin_picking import BinPicking

import rospy, tf
import sys
import os
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from PIL import Image as img_PIL
from pathlib import Path
from bokeh.io import export_png
from bokeh.plotting import gridplot
from happy_service_call import service_call
from multiprocessing import Process, Queue
from get_poses_from_tf import get_poses

from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer import Panda3dLightData
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.utils.conversion import convert_scene_observation_to_panda3d
from happypose.toolbox.visualization.bokeh_plotter import BokehPlotter
from happypose.toolbox.visualization.utils import make_contour_overlay
from happypose.toolbox.utils.logging import get_logger, set_logging_level

from agimus_demos.tools_hpp import concatenatePaths
from agimus_demos.calibration.play_path import CalibrationControl, playAllPaths

logger = get_logger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["EGL_VISIBLE_DEVICES"] = "-1"

print("[START]")
print("To avoid crash during constrain graph building, kill the hppcorbaserver process once in a while.")

connectedToRos = True

# try:
#     from constrain_graph import constrain_graph_variables
# except:
#     print("Failed to import constrain graph variables")

try:
    import rospy
    Robot.urdfString = rospy.get_param('robot_description')
    print("reading URDF from ROS param")
    connectedToRos = True
except:
    print("reading generic URDF")
    from hpp.rostools import process_xacro, retrieve_resource
    Robot.urdfString = process_xacro\
      ("package://agimus_demos/franka/manipulation/urdf/demo.urdf.xacro")
Robot.srdfString = ""

class Box:
    urdfFilename="package://agimus_demos/franka/manipulation/urdf/big_box.urdf"
    srdfFilename="package://agimus_demos/franka/manipulation/srdf/big_box.srdf"
    rootJointType = "freeflyer"

class TLess:
    urdfFilename = \
        "package://agimus_demos/franka/manipulation/urdf/t-less/obj_01.urdf"
    srdfFilename = \
        "package://agimus_demos/franka/manipulation/srdf/t-less/obj_01.srdf"
    rootJointType = "freeflyer"

defaultContext = "corbaserver"
loadServerPlugin(defaultContext, "manipulation-corba.so")
loadServerPlugin(defaultContext, "bin_picking.so")
newProblem()

robot = Robot("robot", "pandas", rootJointType="anchor")
robot.opticalFrame='camera_color_optical_frame'
shrinkJointRange(robot, [f'pandas/panda2_joint{i}' for i in range(1,8)],0.95)
ps = ProblemSolver(robot)

ps.addPathOptimizer("EnforceTransitionSemantic")
ps.addPathOptimizer("SimpleTimeParameterization")
ps.setParameter('SimpleTimeParameterization/order', 2)
ps.setParameter('SimpleTimeParameterization/maxAcceleration', .5)
ps.setParameter('SimpleTimeParameterization/safety', 0.95)

# Add path projector to avoid discontinuities
ps.selectPathProjector ("Progressive", .05)
ps.selectPathValidation("Graph-Progressive", 0.01)
vf = ViewerFactory(ps)
vf.loadObjectModel (TLess, "part")
vf.loadObjectModel (Box, "box")

robot.setJointBounds('part/root_joint', [-1., 1.5, -1., 1., 0., 2.2])
robot.setJointBounds('box/root_joint', [-1., 1., -1., 1., 0., 1.8])

print("Part and box loaded")

robot.client.manipulation.robot.insertRobotSRDFModel\
    ("pandas", "package://agimus_demos/franka/manipulation/srdf/demo.srdf")

# Remove collisions between object and self collision geometries
srdfString = '<robot name="demo">'
for i in range(1,8):
    srdfString += f'<disable_collisions link1="panda2_link{i}_sc" link2="part/base_link" reason="handled otherwise"/>'
srdfString += '<disable_collisions link1="panda2_hand_sc" link2="part/base_link" reason="handled otherwise"/>'
srdfString += '</robot>'
robot.client.manipulation.robot.insertRobotSRDFModelFromString(
    "pandas",  srdfString)

# Discretize handles
ps.client.manipulation.robot.addGripper("pandas/support_link", "goal/gripper1",
    [1.05, 0.0, 1.02,0,-sqrt(2)/2,0,sqrt(2)/2], 0.0)
ps.client.manipulation.robot.addGripper("pandas/support_link", "goal/gripper2",
    [1.05, 0.0, 1.02,0,-sqrt(2)/2,0,sqrt(2)/2], 0.0)
ps.client.manipulation.robot.addHandle("part/base_link", "part/center1",
    [0,0,0,0,sqrt(2)/2,0,sqrt(2)/2], 0.03, 3*[True] + [False, True, True])
ps.client.manipulation.robot.addHandle("part/base_link", "part/center2",
    [0,0,0,0,-sqrt(2)/2,0,sqrt(2)/2], 0.03, 3*[True] + [False, True, True])

# Lock gripper in open position.
ps.createLockedJoint('locked_finger_1', 'pandas/panda2_finger_joint1', [0.035])
ps.createLockedJoint('locked_finger_2', 'pandas/panda2_finger_joint2', [0.035])
ps.setConstantRightHandSide('locked_finger_1', True)
ps.setConstantRightHandSide('locked_finger_2', True)

handles = list()
handles += ["part/lateral_top_%03i"%i for i in range(16)]
handles += ["part/lateral_bottom_%03i"%i for i in range(16)]
handles += ["part/top_%03i"%i for i in range(16)]
handles += ["part/bottom_%03i"%i for i in range(16)]

binPicking = BinPicking(ps)
binPicking.objects = ["part", "box"]
binPicking.robotGrippers = ['pandas/panda2_gripper']
binPicking.goalGrippers = ['goal/gripper1', 'goal/gripper2']
binPicking.goalHandles = ["part/center1", "part/center2"]
binPicking.handles = handles
binPicking.graphConstraints = ['locked_finger_1', 'locked_finger_2']
print("Building constraint graph")
binPicking.buildGraph()

q0 = [0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4, 0.035, 0.035,
      0, 0, 1.2, 0, 0, 0, 1,
      0, 0, 0.761, 0, 0, 0, 1]

render = False # default value

# Create effector
print("Building effector.")
binPicking.buildEffectors([ f'box/base_link_{i}' for i in range(5) ], q0)

print("Generating goal configurations.")
binPicking.generateGoalConfigs(q0)

#___________________from_run_inference_on_exemple_COSYPOSE___________________

def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (example_dir / "meshes").iterdir()
    print(object_dirs)
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset

def rendering(poses):
    data_dir = os.getenv("MEGAPOSE_DATA_DIR")
    example_dir = Path(data_dir) / "examples/tless"

    rgb = np.array(img_PIL.open("camera_view.png"), dtype=np.uint8)
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())
    camera_data.resolution = rgb.shape[:2]

    object_dataset = make_object_dataset(example_dir)
    camera_data.TWC = Transform(np.eye(4))
    renderer = Panda3dSceneRenderer(object_dataset)
    # Data necessary for image rendering
    print(type(poses[0]))
    print(type(poses[0].numpy()))
    print(len(poses))
    print("poses[0].shape : ", poses[0].shape)
    print("poses.size() : ", poses.size())
    print("poses[0].size() : ", poses[0].size())
    object_datas = [ObjectData(label="tless", TWO=Transform(poses[0].numpy()))]
    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((0.6, 0.6, 0.6, 1)),
        ),
    ]
    renderings = renderer.render_scene(
        object_datas,
        [camera_data],
        light_datas,
        render_depth=False,
        render_binary_mask=False,
        render_normals=False,
        copy_arrays=True,
    )[0]
    return renderings

def save_predictions(renderings):
    data_dir = os.getenv("MEGAPOSE_DATA_DIR")
    example_dir = Path(data_dir) / "examples/tless"
    rgb_render = renderings.rgb
    rgb = np.array(img_PIL.open("camera_view.png"), dtype=np.uint8)
    mask = ~(rgb_render.sum(axis=-1) == 0)
    alpha = 0.1
    rgb_n_render = rgb.copy()
    rgb_n_render[mask] = rgb_render[mask]

    print("save prediction")

    # make the image background a bit fairer than the render
    rgb_overlay = np.zeros_like(rgb_render)
    rgb_overlay[~mask] = rgb[~mask] * 0.6 + 255 * 0.4
    rgb_overlay[mask] = rgb_render[mask] * 0.8 + 255 * 0.2
    plotter = BokehPlotter()

    fig_rgb = plotter.plot_image(rgb)

    fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
    contour_overlay = make_contour_overlay(
        rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
    )["img"]
    fig_contour_overlay = plotter.plot_image(contour_overlay)
    fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)
    vis_dir = example_dir / "visualizations_hpp"
    vis_dir.mkdir(exist_ok=True)
    export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay.png")
    export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay.png")
    export_png(fig_all, filename=vis_dir / "all_results.png")
    print("Images save in : ",vis_dir)

#____________________________________________________________________________


def GrabAndDrop(robot, ps, binPicking, render):
    # Get configuration of the robot
    ri = None

    if connectedToRos:
        ri = RosInterface(robot)
        q_init = ri.getCurrentConfig(q0)
        res, q_init, err = binPicking.graph.applyNodeConstraints('free', q_init)
        assert(res)
    else:
        q_init = q0[:]

    # Detecting the object poses
    found = False
    essaie = 0
    q_init, wMo = ri.getObjectPose(q_init)

    poses = np.array(q_init[9:16])
    rotation_matrix = np.array(wMo)
    transformation_matrix = np.zeros((4,4))
    transformation_matrix[:3,:3] = rotation_matrix[:3,:3]
    transformation_matrix[:3,3] = poses[4:7]
    transformation_matrix[3,3] = 1

    print("\nPose of the object : \n",poses,"\n")
    # print("\n Transformation matrix : \n",transformation_matrix,"\n")

    if render:
        print("Rendering the detection on image ...")
        transformation_matrix = torch.tensor([transformation_matrix])
        renderings = rendering(transformation_matrix)
        save_predictions(renderings)
        print("Render finished !")

    while not found and essaie < 25:
        found, msg = robot.isConfigValid(q_init)
        essaie += 1

    # Resolving the path to the object
    if found:
        print("[INFO] Object found with no collision")
        print("Solving ...")
        res = False
        res, p = binPicking.solve(q_init)
        if res:
            ps.client.basic.problem.addPath(p)
            print("Path generated.")
        else:
            print(p)
        return q_init, p

    else:
        print("[INFO] Object found but not collision free")
        print("Trying solving without playing path for simulation ...")

    return q_init

def multiple_GrabAndDrop():
    print("Begining of bin picking.")

    # Cleaning the path vector
    clean_path_vector()

    # Getting the number of object
    print("[INFO] Retriving the number of objects ...")
    nb_obj = service_call()
    print(nb_obj,"objects detected by the service")
    verif = input("What is the exact number of objects : ")

    if verif == '': pass
    elif int(verif) != nb_obj: nb_obj = verif

    # Initialize variables
    path_id = 0
    essaie = 0
    found = False
    ri = RosInterface(robot)
    q_init = ri.getCurrentConfig(q0)
    res, q_init, err = binPicking.graph.applyNodeConstraints('free', q_init)

    for i in range(nb_obj):

        # Starting ROS cosypose detection process
        ros_process = Process(target=service_call)
        ros_process.start()
        
        q_init, wMo = ri.getObjectPose(q_init)
        
        while wMo != None and not found and essaie < 25:
            found, msg = robot.isConfigValid(q_init)
            essaie += 1
        ros_process.terminate()

        # Resolving the path to the object
        if found:
            print("[INFO] Object found with no collision")
            print("Solving ...")
            res = False
            res, p = binPicking.solve(q_init)
            if res:
                ps.client.basic.problem.addPath(p)
                print("Path generated.")
            else:
                print(p)

        else:
            print("[INFO] Object found but not collision free")
            print("Trying solving without playing path for simulation ...")

        cc = CalibrationControl("panda2_hand","camera_color_optical_frame","panda2_ref_camera_link")
        nbPaths = cc.hppClient.problem.numberPaths()
        print("Number of path :",nbPaths)
        
        viz = input("Do you want to vizualise the mouvement first ? ")
        if viz == 'yes':
            v = vf.createViewer()
            v(q_init)
            print("[INFO] Viewer created.")
            input("Press Enter to continue ...")

        # Playing the movement
        print("Starting movement number ",i)
        input("Press Enter to start the movement ...")
        cc.playPath(path_id,collect_data = False)
        if not cc.errorOccured:
            print("Ran {}".format(i))
            i+=1

        print("Once the path is played, press ENTER.")
        input("Press Enter to continue ...")

        path_id += 1

    print("All path played.")

    # Cleaning the path vector (again)
    clean_path_vector()

    return q_init

def simultanous_Grasp():
    print("Begining of simultanous grasp.")
    
    found = False
    essaie = 0
    ri = RosInterface(robot)
    q_init = ri.getCurrentConfig(q0)
    res, q_init, err = binPicking.graph.applyNodeConstraints('free', q_init)

    ros_process = Process(target=service_call)
    ros_process.start()

    q_init, wMo = ri.getObjectPose(q_init)

    while wMo != None and not found and essaie < 25:
        found, msg = robot.isConfigValid(q_init)
        essaie += 1

    # Resolving the path to the object
    if found:
        print("[INFO] Object found with no collision")
        print("Solving ...")
        res = False
        res, p = binPicking.solve(q_init)
        if res:
            ps.client.basic.problem.addPath(p)
            print("Path generated.")
        else:
            print(p)

    else:
        print("[INFO] Object found but not collision free")
        print("Trying solving without playing path for simulation ...")

    return q_init, p

def precise_Grasp():
    print("Begining of precise grasp.")

    ri = RosInterface(robot)
    q_init = ri.getCurrentConfig(q0)
    res, q_init, err = binPicking.graph.applyNodeConstraints('free', q_init)
    q_init, wMo = ri.getObjectPose(q_init)

    v = vf.createViewer()
    v(q_init)

    print("[INFO] Object found with no collision")
    print("Solving ...")
    res = False
    res, p = binPicking.solve(q_init, 'direct_path')
    if res: 
        ps.client.basic.problem.addPath(p)
        print("Path for approach is generated.")
    else:
        print(p)

    print("\nIf the path wasn't generated or the object wasn't detected correctly, you can enter |retry|.")
    print("If the path visualization is missing on gepetto-gui, you can enter |q_init|.")
    print("If you want to exit the function, you can enter |n|.")
    print("If the path was generated correctly and you want to proceed, execute the approach then press |y|")
    confirm = input("Input : ")
    if confirm == 'n':
        print("Exit ...")
        return 0
    if confirm == 'retry':
        precise_Grasp()
    if confirm == 'q_init':
        v = vf.createViewer()
        v(q_init)
        confirm = 'y'
    if confirm == 'y':
        # q_init, p = GrabAndDrop(robot, ps, binPicking, render)
        found = False
        essaie = 0
        q_init = ri.getCurrentConfig(q0)
        res, q_init, err = binPicking.graph.applyNodeConstraints('free', q_init)
        q_init, wMo = ri.getObjectPose(q_init)
        q_init[11] += 0.02 # height +2 cm

        while not found and essaie < 25:
                found, msg = robot.isConfigValid(q_init)
                essaie += 1

        # Resolving the path to the object
        if found:
            print("[INFO] Object found with no collision")
            print("Solving ...")
            res = False
            res, p = binPicking.solve(q_init)
            if res: 
                ps.client.basic.problem.addPath(p)
                print("Path generated.")
            else:
                print(p)

        else:
            print("[INFO] Object found but not collision free")
            print("Trying solving without playing path for simulation ...")

#______________________________Utility_funtions______________________________

def benchmark_pandas():
    keep_moving = True

    while keep_moving:
        ri = RosInterface(robot)
        q_init = ri.getCurrentConfig(q0)
        res, q_init, err = binPicking.graph.applyNodeConstraints('free', q_init)

        """
        q0 = [0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4, 0.035, 0.035,
        0, 0, 1.2, 0, 0, 0, 1,
        |id = 9
        0, 0, 0.761, 0, 0, 0, 1]
                |id = 18

        gripper is 0.125m above the object referential (e.g. obj Z = 1.2 / gripper Z = 1.325)
        gripper coordinate in obj referential :
        x_gripper = -x_obj
        y_gripper = y_obj   
        z_gripper = z_obj + 0.125

        for a 90° rotation of the object :
        q_init[13] = 0.70710678
        q_init[15] = 0.70710678

        When the object_tless_01 is on a surface, its center is 17.495 mm above the surface

        to have the gripper 10 cm above the object -> z = z_obj + 0.09875
        to have the gripper 10 cm above the surface -> z = 0.081255
        to have the gripper x cm above the surface -> x = x - 0.081255
        to have the camera x cm above the object -> x = x- 0.081255 - 0.175
        """

        distance = int(input("How many cm do you want the gripper to have above the object ? ")) # Distance in cm
        distance_from_table = distance/100 - 0.09875 - 0.175 + 0.76  # Distance of the object from the table
        q_init[11] = distance_from_table

        found, msg = robot.isConfigValid(q_init)

        if found:
            print("[INFO] Object found with no collision")
            print("Solving ...")
            res = False
            res, p = binPicking.solve(q_init, 'direct_path')
            if res:
                ps.client.basic.problem.addPath(p)
                print("Path generated.")
            else:
                print(p)

        else:
            print("[INFO] Object found but not collision free")
            print("Returning goal configuration for simulation ...")
        
        motion = input("Do you want to play the movement ? [y/n] : ")
        if motion == 'y':
            move_robot()

        capture = input("Do you want to capture the camera POV ? [y/n/alt] :")
        if capture == 'y':
            try:
                capture_camera(10)
            except:
                print("[WARN] The camera channel might not be on the port 2. Try with a different port.")
                wrong_port = True
                while wrong_port:
                    cam_id = int(input("Channel : "))
                    try:
                        capture_camera(10, cam_id)
                        wrong_port = False
                    except:
                        print("Wrong port again. If you see this message too much, the error might be something else.")
                        print("If the error says that the camera cannot be open by index, close all the usb camera process (eg. roslaunch realsense2_camera and retry.")
                        again = input("Try again ? [y/n] : ")
                        if again == 'n':
                            wrong_port = False

        if capture == 'alt':
            try:
                capture_camera_alt(10)
            except:
                print("[WARN] The camera channel might not be on the port 2. Try with a different port.")
                wrong_port = True
                while wrong_port:
                    cam_id = int(input("Channel : "))
                    try:
                        capture_camera_alt(10, cam_id)
                        wrong_port = False
                    except:
                        print("Wrong port again. If you see this message too much, the error might be something else.")
                        print("If the error says that the camera cannot be open by index, close all the usb camera process (eg. roslaunch realsense2_camera and retry.")
                        again = input("Try again ? [y/n] : ")
                        if again == 'n':
                            wrong_port = False

        cosypose = input("Do you want to run Cosypose ? [y/n] : ")
        if cosypose == 'y':
            run_cosypose()

        erase = input("Erasing the path vector ? [y/n] : ")
        if erase == 'y':
            clean_path_vector()
        
        moving = input("Do you want to move the robot again ? [y/n] : ")
        if moving == 'n':
            keep_moving = False

    return q_init

def move_robot():
    path_id = ps.numberPaths()
    cc = CalibrationControl("panda2_hand","camera_color_optical_frame","panda2_ref_camera_link")
    input("Press Enter to start the movement ...")
    cc.playPath(path_id - 1,collect_data = False)
    if not cc.errorOccured:
        print("Ran {}".format(path_id))

def clean_path_vector():
    # Cleaning the path vector
    number_of_path = ps.numberPaths() #getting the number of path
    print("Number of paths : ",number_of_path)
    if number_of_path > 0:
        print("Cleaning the path vector.")
        # Erasing every vector in the vector path
        for i in range(number_of_path):
            ps.erasePath(number_of_path - i - 1)
            sys.stdout.write("[INFO] Erasing path number %d.\n" % (number_of_path - i - 1))
            sys.stdout.flush()
    else:
        print("No path to clean.")

def capture_camera(nb_cap = 1, cam_id = 2, distance = ""):
    print("[INFO] Opening camera channel on port",cam_id,".")
    cam = cv2.VideoCapture(cam_id) # default camera port is 2
    dir_path = os.getcwd() + '/temp'
    print("To close camera stream, press Q.")
    for k in range(nb_cap):
        result, image = cam.read()
        name_img = str("image_"+str(k)+"_"+str(distance)+".png")
        name = str(dir_path+"/"+name_img)
        if result:
            cv2.imshow("camera capture", image)
            cv2.imwrite(name, image)
            cv2.waitKey(0)
            cv2.destroyWindow("camera capture")
            print("Image captured. Saved under %s as %s." %(dir_path,name_img))
        else:
            print("Error while capturing the image.")

def capture_camera_alt(nb_cap = 1, cam_id = 2):
    print("[INFO] Opening camera channel on port",cam_id,".")
    cam = cv2.VideoCapture(cam_id) # default camera port is 2
    dir_path = os.getcwd() + '/temp'
    print("capture_camera_alt is running on realsense ros. Please assure that 'roslaunch realsense2_camera rs_camera.launch' is running.")
    try:
        rospy.init_node("inference_on_camera_image", anonymous=True)
        for k in range(nb_cap):
            name_img = str("image_"+str(k)+".png")
            name = str(dir_path+"/"+name_img)
            print("Waiting to capture image from camera stream")
            image_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
            bridge = CvBridge()
            image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
            cv2.imwrite(name, image)
            print("Image captured. Saved under %s as %s." %(dir_path,name_img))

    except:
        print("Error while capturing the image.")

def run_cosypose():
    nb_obj = service_call()
    print(nb_obj,"objects detected by the service")

    essaie = 0
    found = False
    ri = RosInterface(robot)
    q_init = ri.getCurrentConfig(q0)
    res, q_init, err = binPicking.graph.applyNodeConstraints('free', q_init)
    filepath = os.getcwd() + '/temp'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    file = open(filepath + '/poses.txt','a')

    for i in range(nb_obj):
        # Starting ROS cosypose detection process
        ros_process = Process(target=service_call)
        ros_process.start()
        
        q_init, wMo = ri.getObjectPose(q_init)

        ros_process.terminate()

        poses = np.array(q_init[9:16])
        rotation_matrix = np.array(wMo)
        transformation_matrix = np.zeros((4,4))
        transformation_matrix[:3,:3] = rotation_matrix[:3,:3]
        transformation_matrix[:3,3] = poses[4:7]
        transformation_matrix[3,3] = 1

        print("\nPose of the object : \n",poses,"\n")
        # print("\n Transformation matrix : \n",transformation_matrix,"\n")

        file.write(str(poses)+"\n")

    file.close()
     
def detect():
    print("[INFO] Getting objects poses.")
    dict_of_poses, list_of_names = get_poses(False)
    list_of_poses = dict_to_list_poses(dict_of_poses,list_of_names)
    write_poses_in_file(list_of_poses)

def dict_to_list_poses(dict_of_poses,list_of_name):
    list_of_poses = []
    list_var = ['x','y','z','theta x','theta y','theta z','theta w']
    nb_of_obj = len(dict_of_poses)
    for i in range(nb_of_obj):
        list_of_poses.append([])
        for j in range(7):
            list_of_poses.append(dict_of_poses[list_of_name[i]][list_var[j]])
    return list_of_poses

def write_poses_in_file(list):
    if len(list) > 0:
        filepath = os.getcwd() + '/temp'
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        file = open(filepath + '/poses.txt','a')
        file.write(str(list))
        file.close()

#____________________________________________________________________________


if __name__ == '__main__':
    print("Script HPP ready !")
    q_start = RosInterface(robot).getCurrentConfig(q0)
    # q_init, p = GrabAndDrop(robot, ps, binPicking, render)
    # nb_obj, poses, infos = happypose_with_camera.get_nb_objects_in_image(0)