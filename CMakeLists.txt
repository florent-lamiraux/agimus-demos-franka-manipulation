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

INSTALL_DEMO_LAUNCH_FILES(franka manipulation
  LAUNCHFILES
  demo.launch
  robot.launch
  rosparam.launch
  simulation.launch
  )

INSTALL(FILES
  urdf/big_box.urdf
  urdf/demo.urdf.xacro
  urdf/franka_arm.xacro
  urdf/franka_hand.xacro
  urdf/franka_robot.xacro
  urdf/support.urdf.xacro
  urdf/utils.xacro
  DESTINATION
  ${CATKIN_PACKAGE_SHARE_DESTINATION}/franka/manipulation/urdf
  )

INSTALL(FILES
  urdf/t-less/obj_01.urdf
  DESTINATION
  ${CATKIN_PACKAGE_SHARE_DESTINATION}/franka/manipulation/urdf/t-less
  )

INSTALL(FILES
  srdf/big_box.srdf
  srdf/demo.srdf
  DESTINATION
  ${CATKIN_PACKAGE_SHARE_DESTINATION}/franka/manipulation/srdf
  )

INSTALL(FILES
  srdf/t-less/obj_01.srdf
  DESTINATION
  ${CATKIN_PACKAGE_SHARE_DESTINATION}/franka/manipulation/srdf/t-less
  )

INSTALL(FILES
  srdf/t-less/obj_01.srdf
  DESTINATION
  ${CATKIN_PACKAGE_SHARE_DESTINATION}/franka/manipulation/srdf/t-less
  )

INSTALL(FILES
  config/calibrated-params.yaml
  DESTINATION
  ${CATKIN_PACKAGE_SHARE_DESTINATION}/franka/manipulation/config
  )

INSTALL(FILES
  supervisor.py
  demo.yaml
  demo-calibration.yaml
  DESTINATION
  ${CATKIN_PACKAGE_SHARE_DESTINATION}/franka/manipulation
  )

INSTALL(FILES
  meshes/camera-support.stl
  DESTINATION
  ${CATKIN_PACKAGE_SHARE_DESTINATION}/franka/manipulation/meshes
  )

INSTALL(DIRECTORY
  meshes/t-less DESTINATION
  ${CATKIN_PACKAGE_SHARE_DESTINATION}/franka/manipulation/meshes
  )

INSTALL(PROGRAMS
  scripts/gripper_control.py
  DESTINATION
  ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
  RENAME franka_manipulation_gripper_control.py
  )
