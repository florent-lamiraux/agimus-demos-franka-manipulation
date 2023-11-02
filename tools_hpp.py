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

import numpy as np
import os
import rospy, tf2_ros
import numpy
from hpp import Transform
from math import sqrt, pi
from pinocchio import XYZQUATToSE3, SE3ToXYZQUAT
from agimus_demos.tools_hpp import RosInterface as Parent

## Randomly shoot a configuration where the part is in the box
def shootPartInBox(robot, q0):
    robot.setCurrentConfig(q0)
    r = robot.rankInConfiguration['box/root_joint']
    Mb = Transform(q0[r:r+7])
    q = robot.shootRandomConfig()
    r = robot.rankInConfiguration['part/root_joint']
    q[r+0] = .56 * np.random.uniform() - .28
    q[r+1] = .36 * np.random.uniform() - .18
    q[r+2] = .20 * np.random.uniform()
    Mp = Mb * Transform(q[r:r+7])
    qres = q0[:]
    qres[r:r+7] = list(Mp)
    return qres

## Display a frame in gepetto-gui corresponding to a handle
# \param robot instance of class hpp.corbaserver.robot.Robot
# \param name name of the handle.
def displayHandle(viewer, name):
    robot = viewer.robot
    joint, pose = robot.getHandlePositionInJoint(name)
    hname = 'handle__' + name.replace('/', '_')
    viewer.client.gui.addXYZaxis(hname, [0, 1, 0, 1], 0.005, 0.015)
    if joint != "universe":
        link = robot.getLinkNames(joint)[0]
        viewer.client.gui.addToGroup(hname, robot.name + '/' + link)
    else:
        viewer.client.gui.addToGroup(hname, robot.name)
    viewer.client.gui.applyConfiguration(hname, pose)

## Display a frame in gepetto-gui corresponding to a handle
# \param robot instance of class hpp.corbaserver.robot.Robot
# \param name name of the handle.
def displayGripper(viewer, name):
    robot = viewer.robot
    joint, pose = robot.getGripperPositionInJoint(name)
    gname = 'gripper__' + name.replace('/', '_')
    viewer.client.gui.addXYZaxis(gname, [0, 1, 0, 1], 0.005, 0.015)
    if joint != "universe":
        link = robot.getLinkNames(joint)[0]
        viewer.client.gui.addToGroup(gname, robot.name + '/' + link)
    else:
        viewer.client.gui.addToGroup(gname, robot.name)
    viewer.client.gui.applyConfiguration(gname, pose)

# Generate target config from randomly sampled configurations
# Default Nsample = 20
# len(binPicking.goalConfigs['preplace']['pandas/panda2_gripper'].keys())

def generateTargetConfig(robot, graph, edge, q, Nsamples = 50):
    res = False
    minErr = 1e8
    for i in range(Nsamples + 1):
        if i == 0:
            qrand = q[:]
        else:
            qrand = robot.shootRandomConfig()
            r = robot.rankInConfiguration['box/root_joint']
            qrand[r:r+7] = q[r:r+7]
        res, q1, err = graph.generateTargetConfig(edge, q, qrand)
        if err < minErr:
            minErr = err
            best_q = q1[:]
        if not res: continue
        res, msg = robot.isConfigValid(q1)
        if res: break
    if res:
        return True, q1
    else:
        return False, best_q

class RosInterface(Parent):
    def getObjectPose(self, q0, timeout=5):
        # the object should be placed wrt to the robot, as this is what the
        # sensor tells us.
        # Get pose of object wrt to the camera using TF
        cameraFrame = self.robot.opticalFrame
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        qres = q0[:]

        found  = False

        objectFrame_1 = 'object_tless_1_1'
        objectFrame_2 = 'object_tless_2_1'

        print("[INFO] Poses not published yet. Waiting for pose to be published.")
        print("[INFO] You need to launch the 'happypose_inference' service.")
        print("...")
        
        while not found:
            if self.tfBuffer.can_transform(cameraFrame, objectFrame_1, rospy.Time()):
                obj_found = objectFrame_1
                found = True
                print("Object Tless 1 found.")
            if self.tfBuffer.can_transform(cameraFrame, objectFrame_2, rospy.Time()):
                obj_found = objectFrame_2
                found = True
                print("Object Tless 2 found.")
            rospy.sleep(0.01)
        
        print("[INFO] Pose found !")
        print("... Starting calculating transform in the camera frame land mark ...")
        wMc = XYZQUATToSE3(self.robot.hppcorba.robot.getJointsPosition(q0, [self.robotPrefix + cameraFrame])[0])
        print(f"wMc = {wMc}")
        try:
            _cMo = self.tfBuffer.lookup_transform(cameraFrame, obj_found, rospy.Time(), rospy.Duration(timeout))
            _cMo = _cMo.transform
            # renormalize quaternion
            x = _cMo.rotation.x
            y = _cMo.rotation.y
            z = _cMo.rotation.z
            w = _cMo.rotation.w
            n = sqrt(x*x+y*y+z*z+w*w)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            print('could not get TF transform : ', e)
            raise RuntimeError(str(e))
        cMo = XYZQUATToSE3([_cMo.translation.x, _cMo.translation.y,
                            _cMo.translation.z, _cMo.rotation.x/n,
                            _cMo.rotation.y/n, _cMo.rotation.z/n,
                            _cMo.rotation.w/n])
        print(f"cMo = {cMo}")
        rk = self.robot.rankInConfiguration['part/root_joint']
        assert self.robot.getJointConfigSize('part/root_joint') == 7
        wMo = wMc * cMo
        print(f"wMo = {wMc * cMo}")
        qres[rk:rk+7] = SE3ToXYZQUAT (wMc * cMo)
            
        return qres, wMo
