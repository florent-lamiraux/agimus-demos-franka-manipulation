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
from agimus_demos.tools_hpp import RosInterface
from hpp.corbaserver import wrap_delete
from t_less import TLess
from tools_hpp import displayGripper, displayHandle, generateTargetConfig, \
    shootPartInBox
from bin_picking import BinPicking

connectedToRos = False

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

defaultContext = "corbaserver"
loadServerPlugin(defaultContext, "manipulation-corba.so")
loadServerPlugin(defaultContext, "bin_picking.so")
newProblem()

robot = Robot("robot", "pandas", rootJointType="anchor")
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
part = TLess(vf, name="part", obj_id="01")
vf.loadRobotModel (Box, "box")

robot.setJointBounds('part/root_joint', [-1., 1.5, -1., 1., 0., 2.2])
robot.setJointBounds('box/root_joint', [-1., 1., -1., 1., 0., 1.8])

print("Part and box loaded")

robot.client.manipulation.robot.insertRobotSRDFModel\
    ("pandas", "package://agimus_demos/franka/manipulation/srdf/demo.srdf")

# Discretize handles
handles = list()
ps.client.manipulation.robot.addGripper("pandas/support_link", "goal/gripper1",
    [1.05, 0.4, 1.,0,-sqrt(2)/2,0,sqrt(2)/2], 0.0)
ps.client.manipulation.robot.addGripper("pandas/support_link", "goal/gripper2",
    [1.05, 0.5, 1.,0,-sqrt(2)/2,0,sqrt(2)/2], 0.0)
ps.client.manipulation.robot.addHandle("part/base_link", "part/center1",
    [0,0,0,0,sqrt(2)/2,0,sqrt(2)/2], 0.03, 3*[True] + [False, True, True])
ps.client.manipulation.robot.addHandle("part/base_link", "part/center2",
    [0,0,0,0,-sqrt(2)/2,0,sqrt(2)/2], 0.03, 3*[True] + [False, True, True])

# Lock gripper in open position.
ps.createLockedJoint('locked_finger_1', 'pandas/panda2_finger_joint1', [0.035])
ps.createLockedJoint('locked_finger_2', 'pandas/panda2_finger_joint2', [0.035])
ps.setConstantRightHandSide('locked_finger_1', True)
ps.setConstantRightHandSide('locked_finger_2', True)

binPicking = BinPicking(ps)
binPicking.objects = ["part", "box"]
binPicking.robotGrippers = ['pandas/panda2_gripper']
binPicking.goalGrippers = ['goal/gripper1', 'goal/gripper2']
binPicking.goalHandles = ["part/center1", "part/center2"]
binPicking.handlesToDiscretize = ['part/lateral_top', 'part/lateral_bottom',
                                  'part/top', 'part/bottom' ]
binPicking.graphConstraints = ['locked_finger_1', 'locked_finger_2']
print("Building constraint graph")
binPicking.buildGraph()

q0 = [0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4, 0.035, 0.035,
      0, 0, 1.2, 0, 0, 0, 1,
      0, 0, 0.761, 0, 0, 0, 1]

if connectedToRos:
    ri = RosInterface(robot)
    q = ri.getCurrentConfig(q0)
else:
    q = q0[:]

# Create effector
print("Building effector.")
binPicking.buildEffectors([ f'box/base_link_{i}' for i in range(5) ], q0)

print("Generating goal configurations.")
binPicking.generateGoalConfigs(q0)

for i in range(10):
    found = False
    while not found:
        q = shootPartInBox(robot, q0)
        found, msg = robot.isConfigValid(q)
    print("Solving")
    res = False
    res, p = binPicking.solve(q)
    if res:
        ps.client.basic.problem.addPath(p)
