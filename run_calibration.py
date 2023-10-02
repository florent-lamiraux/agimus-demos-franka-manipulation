#!/usr/bin/env python
# Copyright 2021 CNRS - Airbus SAS
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

from agimus_demos.calibration.play_path import CalibrationControl, playAllPaths

if __name__ == '__main__':
    import rospy
    cc = CalibrationControl ()
    cc.endEffectorFrame = "panda2_ref_camera_link"
    cc.mountFrame = "panda2_hand"
    cc.cameraFrame = "camera_color_optical_frame"
    cc.squareSize = 0.0254
    cc.joints = ['panda2_joint1', 'panda2_joint2', 'panda2_joint3',
        'panda2_joint4', 'panda2_joint5', 'panda2_joint6',
        'panda2_joint7', 'panda2_finger_joint1', 'panda2_finger_joint2', ]

    rospy.sleep (5.)
    i = 0
    nbPaths = cc.hppClient.problem.numberPaths ()
    while i < nbPaths - 1:
        cc.playPath (i)
        if not cc.errorOccured:
            print("Ran {}".format(i))
            i+=1
    if nbPaths > 1:
        cc.playPath(nbPaths - 1, collect_data = False)
    cc.save()
    cc.computeHandEyeCalibration()
    eMc = cc.computeCameraPose()
    cc.writeCameraParameters(eMc)
