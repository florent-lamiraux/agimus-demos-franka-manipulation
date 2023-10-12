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

import rospy
import sys

print("[START]")
print("To avoid crash during constrain graph building, kill the hppcorbaserver process once in a while.")

connectedToRos = True

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
    [1.05, -0.05, 1.,0,-sqrt(2)/2,0,sqrt(2)/2], 0.0)
ps.client.manipulation.robot.addGripper("pandas/support_link", "goal/gripper2",
    [1.05, 0.05, 1.,0,-sqrt(2)/2,0,sqrt(2)/2], 0.0)
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

# Create effector
print("Building effector.")
binPicking.buildEffectors([ f'box/base_link_{i}' for i in range(5) ], q0)

print("Generating goal configurations.")
binPicking.generateGoalConfigs(q0)


def GrabAndDrop(robot, ps, binPicking):
    # Get configuration of the robot
    ri = None

    if connectedToRos:
        ri = RosInterface(robot)
        q_init = ri.getCurrentConfig(q0)
        res, q_init, err = binPicking.graph.applyNodeConstraints('free', q_init)
        assert(res)
    else:
        q_init = q0[:]

    # nb_obj, poses, infos = happypose_with_camera.get_nb_objects_in_image(0)

    # Detecting the object poses
    found = False
    essaie = 0
    q_init = ri.getObjectPose(q_init)
    print(q_init)

    while not found and essaie < 25:
        found, msg = robot.isConfigValid(q_init)
        essaie += 1

    # Resolving the path to the object
    if found:
        print("[INFO] Solution found")
        print("Solving ...")
        res = False
        res, p = binPicking.solve(q_init)
        if res:
            ps.client.basic.problem.addPath(p)
        else:
            print(p)

    else:
        print("[INFO] Solution not found")
        print("Trying solving without playing path for simulation ...")
        res = False
        res, p = binPicking.solve(q_init)
        if res:
            ps.client.basic.problem.addPath(p)
        else:
            print(p)
    return q_init, p

if __name__ == '__main__':
    q_init, p = GrabAndDrop(robot, ps, binPicking)
