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
    ConstraintGraphFactory, CorbaClient, SecurityMargins, Constraints
from hpp.gepetto.manipulation import ViewerFactory
from hpp.corbaserver import wrap_delete
from tools_hpp import displayGripper, displayHandle, generateTargetConfig, \
    shootPartInBox, RosInterface
from bin_picking import BinPicking
from logging import getLogger

import rospy, tf2_ros
import sys
import os
import numpy as np
import cv2
import torch
import time
import ast
import bridge_transform as btf
import data_acquisition as dta

from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from PIL import Image as img_PIL
from pathlib import Path
from bokeh.io import export_png
from bokeh.plotting import gridplot
from multiprocessing import Process, Queue
from pyquaternion import Quaternion

from agimus_demos.tools_hpp import concatenatePaths
from agimus_demos.calibration.play_path import CalibrationControl, playAllPaths
from agimus_demos.calibration import HandEyeCalibration as Calibration
from hpp.corbaserver.manipulation import ConstraintGraphFactory as Factory

logger = getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["EGL_VISIBLE_DEVICES"] = "-1"

print("[START]")
print("To avoid crash during constrain graph building, RESTART the hppcorbaserver process once in a while.")

connectedToRos = True

try:
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

def disable_collision():
    srdf_disable_collisions = """<robot>\n"""
    srdf_disable_collisions_fmt = """  <disable_collisions link1="{}" link2="{}" reason=""/>\n"""
    srdf_disable_collisions += srdf_disable_collisions_fmt.format("box/base_link",
                                                                  "part/base_link")
    srdf_disable_collisions += "</robot>"
    robot.client.manipulation.robot.insertRobotSRDFModelFromString("", srdf_disable_collisions)
    print(srdf_disable_collisions)

disable_collision()

print("Building constraint graph")
binPicking.buildGraph()

q0 = [0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4, 0.035, 0.035,
      0, 0, 1.2, 0, 0, 0, 1,
      0, 0, 0.761, 0, 0, 0, 1]

# Create effector
print("Building effector.")
binPicking.buildEffectors([ f'box/base_link_{i}' for i in range(5) ], q0)

print("Generating goal configurations.")
binPicking.generateGoalConfigs(q0)

#___________________________END_OF_GRAPH_GENERATION__________________________

def GrabAndDrop(robot, ps, binPicking, acq_type = None):
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

    #____________GETTING_THE_POSE____________

    # Type of data acquisition
    # test_config
    # input_config
    # ros_bridge_config

    # quaternion is X, Y, Z, W

    if acq_type == 'test_config':
        print("[INFO] Test config.")
        q_sim = [0,0.2, 0.65, 0.2917479872902073, 0.6193081061291802, 0.6618066799607849, 0.30553641346668353]
        q_init, wMo = q_init,None
        print("test")
        q_init[9:16] = q_sim

    if acq_type == 'input_config':
        print("[INFO] Given config.")
        data_input = input("Enter the XYZQUAT : ")
        data_input = ast.literal_eval(data_input)
        quat = Quaternion(data_input[3],data_input[4],data_input[5],data_input[6])
        quat = quat.normalised
        q_input = [data_input[0],data_input[1],data_input[2],quat[0], quat[1], quat[2], quat[3]]
        q_init[9:16], wMo =  q_input, None

    if acq_type == 'ros_bridge_config':
        print("[INFO] Make sure the /happypose/detections ros topic exist !")
        input("Press [ENTER] to proceed ...")
        data = btf.run_pipeline()
        max_Z = data[0].position.z
        id = 0
        for i in range(len(data)):
            if data[i].position.z > max_Z:
                id = i
                max_Z = data[i].position.z
                print(data[i].position.z,">",max_Z)
        quat = Quaternion([data[id].orientation.x, data[id].orientation.y, data[id].orientation.z, data[id].orientation.w])
        quat = quat.normalised
        q_bridge = [data[id].position.x, data[id].position.y, data[id].position.z,quat[0], quat[1], quat[2], quat[3]]
        q_init[9:16], wMo =  q_bridge, None

    if acq_type !='test_config' and acq_type !='input_config' and acq_type !='ros_bridge_config':
        print("[INFO] No config given to the object.")
        q_init, wMo = ri.getObjectPose(q_init)
    #________________________________________

    q_init[9:16] = check_height(q_init[9:16])
    poses = np.array(q_init[9:16])

    print(q_init)
    print("\nPose of the object : \n",poses,"\n")

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

    return q_init,None

#______________________________Utility_funtions______________________________

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

def check_height(poses):
    if poses[2] < 0.775:
        # Force the height of the object if he is found under the box or the table
        poses[2] = 0.7925 #76.5 + 1 + 1.7495 (height of the table + wide of the box + radius of the obj_tless-000001 base)
        print("[INFO] Set the height to 77.5 cm.")
    return poses

def multiview_data_acquisition(nb=10):
    print("The script will run",nb,"times. For each iteration, move the robot to the desired configuration then press ENTER.")
    for k in range(nb):
        input("Press ENTER to proceed with the data acquisition "+ str(k+1) +" ...")
        cam_pose_quat = dta.get_cam_pose()
        img = dta.capture_camera_image()
        list_of_cam_pose[k] = dta.quat2SE3(cam_pose_quat)[0]
        list_of_images[k] = img[0]
        q = ri.getCurrentConfig(q_start)
        list_of_q[k] = q[0:7]
    
    if not os.path.exists('multiview_data'):
        os.makedirs('multiview_data')
    
    print("Data saved in",os.getcwd()+'/multiview_data')
    np.save("multiview_data/color_img.npy",list_of_images)
    np.save("multiview_data/cam_poses.npy",list_of_cam_pose)
    np.save("multiview_data/q.npy",list_of_q)

def add_tless_to_scene(name, poses):
    from gepetto.corbaserver import Client

    c = Client()
    c.gui.addMesh(name, '/home/dbaudu/devel/src/agimus-demos/franka/manipulation/urdf/t-less/obj_01.urdf')
    c.gui.addToGroup(name, 'scene_hpp_')
    c.gui.setScale(name, [0.001, 0.001, 0.001])
    c.gui.applyConfiguration(name, poses)
    print("[INFO] Add object", name, "to the scene.")

    c.gui.refresh()

#____________________________________________________________________________


if __name__ == '__main__':
    print("Script HPP ready !")
    q_start = RosInterface(robot).getCurrentConfig(q0)
    ri = RosInterface(robot)

    list_of_cam_pose= np.zeros(shape=(10,4,4))
    list_of_images = np.zeros(shape=(10,720,1280,3),dtype=np.uint8)
    list_of_q = np.zeros(shape=(10,7))

    default_acq_type = 'ros_bridge_config'

    # Type of data acquisition
    test_config = 'test_config'
    input_config = 'input_config'
    ros_bridge_config = 'ros_bridge_config'

    # q_init, p = GrabAndDrop(robot, ps, binPicking, 'ros_bridge_config')
