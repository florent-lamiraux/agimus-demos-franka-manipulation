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

import time
import numpy as np
import rospy
from math import sqrt
from dynamic_graph.ros import RosPublish, RosSubscribe
from agimus_sot.action import Action
from franka_gripper.msg import GraspActionGoal, MoveActionGoal
from hpp.corbaserver.manipulation import Rule

Action.maxControlSqrNorm = 40


class OpenGripper(object):
    """
    Preaction that opens the gripper by publishing on ros topic
    /franka_gripper/move/goal
    """
    # Timeout for opening the gripper
    timeout = 5
    # Tolerance to consider that the gripper is open
    eps = 0.001
    def __init__(self, robot, moveActionGoal):
        self.robot = robot
        self.moveActionGoal = moveActionGoal
        self.ros_publish = RosPublish("pub_OpenGripper")
        signal_name = 'sout'
        self.ros_publish.add('vector', signal_name, '/agimus/sot/open_gripper')
        self.signal = self.ros_publish.signal(signal_name)
        # Subscribe to topic that broadcast result of opening and grasping
        self.ros_subscribe = RosSubscribe("pub_GripperSuccess")
        # As we use the same entity for several instances of this class,
        # we need to add the signal only once.
        if not self.ros_subscribe.hasSignal('sout'):
            self.ros_subscribe.add("int", signal_name,
                                   '/agimus/gripper/feedback')

    def __call__(self):
        # Send command
        self.signal.value = np.array([self.moveActionGoal.goal.width,
                                      self.moveActionGoal.goal.speed])
        t = self.robot.device.control.time
        self.ros_publish.signal("trigger").recompute(t)
        # wait for gripper feedback
        self.ros_subscribe.signal('sout').recompute(t)
        res = None
        # Wait for gripper_control to publish 0 in /agimus/gripper/feedback
        while res != 0:
            res = self.ros_subscribe.signal('sout').value
        while res == 0:
            time.sleep(.1)
            t = self.robot.device.state.time
            self.ros_subscribe.signal('sout').recompute(t)
            res = self.ros_subscribe.signal('sout').value
            if res == 1:
                return True, ""
            if res == -1:
                return False, "Failed to open the gripper."
            rospy.loginfo(f"res = {res} is not a regular value.")
        return 0

class CloseGripper(object):
    """
    Preaction that opens the gripper by publishing on ros topic
    /franka_gripper/move/goal
    """
    # Tolerance to consider that the gripper is open
    eps = 0.001
    def __init__(self, robot, graspActionGoal):
        self.robot = robot
        self.graspActionGoal = graspActionGoal
        self.ros_publish = RosPublish("pub_CloseGripper")
        signal_name = 'sout'
        self.ros_publish.add('vector', signal_name, '/agimus/sot/close_gripper')
        self.signal = self.ros_publish.signal(signal_name)
        # Subscribe to topic that broadcast result of opening and grasping
        self.ros_subscribe = RosSubscribe("pub_GripperSuccess")
        # As we use the same entity for several instances of this class,
        # we need to add the signal only once.
        if not self.ros_subscribe.hasSignal('sout'):
            self.ros_subscribe.add("int", signal_name,
                                   '/agimus/gripper/feedback')

    def __call__(self):
        self.signal.value = np.array([self.graspActionGoal.goal.width,
        self.graspActionGoal.goal.epsilon.inner,
        self.graspActionGoal.goal.epsilon.outer,
        self.graspActionGoal.goal.speed,
        self.graspActionGoal.goal.force,])
        t = self.robot.device.control.time
        self.ros_publish.signal("trigger").recompute(t)
        # wait for gripper feedback
        self.ros_subscribe.signal('sout').recompute(t)
        res = None
        # Wait for gripper_control to publish 0 in /agimus/gripper/feedback
        while res != 0:
            res = self.ros_subscribe.signal('sout').value
        while res == 0:
            time.sleep(.1)
            t = self.robot.device.state.time
            self.ros_subscribe.signal('sout').recompute(t)
            res = self.ros_subscribe.signal('sout').value
            if res == 1:
                return True, ""
            if res == -1:
                return False, "Failed to close the gripper."
            rospy.loginfo(f"res = {res} is not a regular value.")
        return 0


def makeSupervisorWithFactory(robot):
    from agimus_sot import Supervisor
    from agimus_sot.factory import Factory, Affordance
    from agimus_sot.srdf_parser import parse_srdf, attach_all_to_link
    import pinocchio
    from rospkg import RosPack
    rospack = RosPack()

    srdf = {}
    # retrieve objects from ros param
    robotDict = globalDemoDict["robots"]
    if len(robotDict) != 1:
        raise RuntimeError("One and only one robot is supported for now.")
    if "objects" in globalDemoDict:
        objectDict = globalDemoDict["objects"]
    else:
        objectDict = dict()

    objects = list(objectDict.keys())
    # parse robot and object srdf files
    srdfDict = dict()
    for r, data in robotDict.items():
        srdfDict[r] = parse_srdf(srdf = data["srdf"]["file"],
                                 packageName = data["srdf"]["package"],
                                 prefix=r)
    for o, data in objectDict.items():
        objectModel = pinocchio.buildModelFromUrdf\
                      (rospack.get_path(data["urdf"]["package"]) + "/" +
                       data["urdf"]["file"])
        srdfDict[o] = parse_srdf(srdf = data["srdf"]["file"],
                                 packageName = data["srdf"]["package"],
                                 prefix=o)
        attach_all_to_link(objectModel, "base_link", srdfDict[o])

    # Add goal grippers
    srdfDict["pandas"]["grippers"]["goal/gripper1"] = {
        "clearance": 0.0, "link": "support_link",
        "position": [1.05, 0.4, 1.,0,-sqrt(2)/2,0,sqrt(2)/2],
        "name": "goal/gripper1", "robot": "pandas"
    }
    srdfDict["pandas"]["grippers"]["goal/gripper2"] = {
        "clearance": 0.0, "link": "support_link",
        "position": [1.05, 0.5, 1.,0,-sqrt(2)/2,0,sqrt(2)/2],
        "name": "goal/gripper2", "robot": "pandas"
    }
    if "part" in srdfDict:
        # Add goal handles
        srdfDict["part"]["handles"]["part/center1"] = {
            "robot": "part", "name": "part/center1", "clearance": 0.03,
            "link": "part/base_link", "position":
            [0,0,0,0,sqrt(2)/2,0,sqrt(2)/2],
            "mask": 3*[True] + [False, True, True]
        }
        srdfDict["part"]["handles"]["part/center2"] = {
            "robot": "part", "name": "part/center2", "clearance": 0.03,
            "link": "part/base_link", "position":
            [0,0,0,0,-sqrt(2)/2,0,sqrt(2)/2],
            "mask": 3*[True] + [False, True, True]
        }

    grippers = list()
    if "grippers" in globalDemoDict:
        grippers = list(globalDemoDict["grippers"])
        grippers.append("goal/gripper1")
        grippers.append("goal/gripper2")

    handlesPerObjects = list()
    contactPerObjects = list()
    for o in objects:
        handlesPerObjects.append(sorted(list(srdfDict[o]["handles"].keys())))
        contactPerObjects.append(sorted(list(srdfDict[o]["contacts"].keys())))

    for w in ["grippers", "handles","contacts"]:
        srdf[w] = dict()
        for k, data in srdfDict.items():
            srdf[w].update(data[w])

    envContactSurfaces = list()
    envContactSurfaces = list(globalDemoDict.get("contact surfaces",[]))
    print (f"env contact surfaces: {envContactSurfaces}")
    supervisor = Supervisor(robot, prefix=list(robotDict.keys())[0])
    factory = Factory(supervisor)
    factory.createPreplaceAnyway = True
    factory.parameters["period"] = 0.01  # TODO soon: robot.getTimeStep()
    factory.parameters["simulateTorqueFeedback"] = simulateTorqueFeedbackForEndEffector
    factory.parameters["addTracerToAdmittanceController"] = False
    # factory.parameters["addTimerToSotControl"] = True
    factory.setGrippers(grippers)
    factory.setObjects(objects, handlesPerObjects, contactPerObjects)
    factory.environmentContacts(envContactSurfaces)

    from hpp.corbaserver.manipulation import Rule
    factory.setupFrames(srdf["grippers"], srdf["handles"], robot)
    if len(handlesPerObjects) > 0:
        for k in handlesPerObjects[0]:
            factory.handleFrames[k].hasVisualTag = False
    factory.setupContactFrames(srdf["contacts"])
    # Read rules if published in ros parameter
    if "rules" in globalDemoDict:
        rules = list()
        for r in globalDemoDict["rules"]:
            rules.append(Rule(grippers=r["grippers"], handles=r["handles"],
                              link=r["link"]))
        factory.setRules(rules)
    factory.generate()

    moveActionGoal = MoveActionGoal()
    moveActionGoal.goal.width = 0.08
    moveActionGoal.goal.speed = 0.1
    openGripper = OpenGripper(robot, moveActionGoal)
    graspActionGoal = GraspActionGoal()
    graspActionGoal.goal.width = 0.01
    graspActionGoal.goal.epsilon.inner = 0.1
    graspActionGoal.goal.epsilon.outer = 0.1
    graspActionGoal.goal.speed = 0.1
    graspActionGoal.goal.force = 5.
    closeGripper = CloseGripper(robot, graspActionGoal)
    ig = 0; g = factory.grippers[ig]
    for ih, h in enumerate(factory.handles):
        # Add preaction to open the gripper
        transitionName_12 = f'{g} > {h} | f_12'
        if transitionName_12 in supervisor.actions:
            supervisor.actions[transitionName_12].preActions.append(openGripper)
            transitionName_23 = f'{g} > {h} | f_23'
            supervisor.actions[transitionName_23].preActions.append(closeGripper)
            
    # Find transitions that release the object to insert gripper opening action
    gripper = grippers[0]
    transitions = list(supervisor.actions.keys())
    # name should start with the name of the robot gripper ("gripper < handle")
    t1 = list(filter(lambda s:s.startswith(gripper), transitions))
    # starting state contains 2 grasps
    t2 = list(filter(lambda s:s.find(':') != -1, t1))
    # first transition that releases the grasp
    t3 = list(filter(lambda s:s.find('_21') != -1, t2))
    for transitionName_21 in t3:
        supervisor.actions[transitionName_21].preActions.append(openGripper)
    supervisor.makeInitialSot()
    return factory, supervisor

factory, supervisor = makeSupervisorWithFactory(robot)

supervisor.plugTopicsToRos()
supervisor.plugSot("")
