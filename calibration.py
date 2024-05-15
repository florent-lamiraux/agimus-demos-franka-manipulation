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

from math import pi, sqrt
from hpp.corbaserver import loadServerPlugin, shrinkJointRange
from hpp.corbaserver.manipulation import Robot, \
    createContext, newProblem, ProblemSolver, ConstraintGraph, \
    ConstraintGraphFactory, Rule, Constraints, CorbaClient, SecurityMargins
from hpp.gepetto.manipulation import ViewerFactory
from agimus_demos.tools_hpp import RosInterface
from agimus_demos.calibration import HandEyeCalibration as Calibration
from hpp.corbaserver import wrap_delete
from hpp.corbaserver.manipulation import ConstraintGraphFactory as Factory
from pinocchio import XYZQUATToSE3, SE3ToXYZQUAT
import tf2_ros
from t_less import TLess
from bin_picking import BinPicking

connectedToRos = False
class CalibrationChessboard:
    urdfFilename = "package://agimus_demos/urdf/chessboard_10x7_27mm.urdf"
    srdfFilename = ""
    rootJointType = "freeflyer"

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

defaultContext = "corbaserver"
loadServerPlugin (defaultContext, "manipulation-corba.so")
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
vf.loadRobotModel (CalibrationChessboard, "part")

robot.setJointBounds('part/root_joint', [-1., 1., -1., 1., -0.8, 1.5])
print("Part loaded")

robot.client.manipulation.robot.insertRobotSRDFModel\
    ("pandas", "package://agimus_demos/franka/manipulation/srdf/demo.srdf")

graph = ConstraintGraph(robot, 'graph')
factory = Factory(graph)
# Add a state in the constraint graph
factory.objects = ["part"]
factory.grippers = list()
factory.handlesPerObjects = [[]]
factory.contactsPerObjects = [[]]
factory.handles = list()
factory.generate()
c = Calibration(ps, graph, factory)
c.robot_name = "pandas"
c.camera_frame = "camera_color_optical_frame"
c.security_distance_robot_universe = 0.05
c.addStateToConstraintGraph()
q0 = [0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4, 0.035, 0.035,
      0.189, 0, 0.761, 0, 0, 0, 1]

# Lock gripper in open position.
ps.createLockedJoint('locked_finger_1', 'pandas/panda2_finger_joint1', [0.035])
ps.createLockedJoint('locked_finger_2', 'pandas/panda2_finger_joint2', [0.035])
ps.setConstantRightHandSide('locked_finger_1', True)
ps.setConstantRightHandSide('locked_finger_2', True)
graph.addConstraints(graph=True,
                     constraints = Constraints(numConstraints =
                        ['locked_finger_1', 'locked_finger_2']))
graph.initialize()


if connectedToRos:
    ri = RosInterface(robot)
    q = ri.getCurrentConfig(q0)
else:
    q = q0[:]

q_init = ri.getCurrentConfig(q0)
res, q_init, err = graph.applyNodeConstraints('free', q_init)
assert res
c.generateConfigurationsAndPaths(q_init, 20)
# c.generateConfigurationsAndPaths(q_init, 30)
