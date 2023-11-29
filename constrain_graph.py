# Copyright 2023 CNRS
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

import rospy
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


def build_constrain_graph():
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
    print("Building constraint graph")
    binPicking.buildGraph()

    global g_robot
    global g_ps
    global g_binPicking
    global g_vf

    g_robot = robot
    g_ps = ps
    g_binPicking = binPicking
    g_vf = vf

    return robot, ps, binPicking, vf

def get_variable():
    robot = g_robot
    ps = g_ps
    binPicking = g_binPicking
    vf = g_vf
    return robot, ps, binPicking, vf

# class constrain_graph_variables:
    # robot, ps, binPicking, vf = build_constrain_graph()

if __name__ == '__main__':
    print("Script constrain graph ready !")
    # robot, ps, binPicking, vf = build_constrain_graph()