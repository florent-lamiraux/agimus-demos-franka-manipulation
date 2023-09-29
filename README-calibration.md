# Instructions to run the hand-eye calibration procedure

This file explains how to
  - plan, execute motions that move the camera in front of a chessboard,
  - collect eye to hand calibration data,
  - compute and save hand eye calibration parameters

1. connect to the ip adress of the robot using firefox. On akasan, this
   address is 172.17.1.3.

2. release the brakes,
3. activate FCI,
4. in terminal 1, in this directory
   ```bash
   roslaunch robot.launch arm_id:=panda2 robot_ip:=172.17.1.3
   ```
5. to start the camera
   ```bash
   roslaunch realsense2_camera rs_camera.launch
   ```

6. to start agimus / SoT
   ```
   roslaunch demo.launch calibration:=true
   ```

7. Start hppcorbaserver
   ```
   hppcorbaserver
   ```

8. Start the GUI
   ```
   gepetto-gui
   ```
9. run `calibration.py` script in a terminal

  ```
  cd agimus-demos/franka/manipulation
  python -i calibration.py
  ```
  This should compute some paths going to configurations where the robot
  looks at the chessboard.

  In the same terminal, display the robot and environment:

  ```
  >>> v = vf.createViewer()
  ```

10. place the chessboard as shown in `gepetto-gui`.

11. in a new terminal,

  ```
  python -i run_calibration.py
  ```
  This should
    - execute the motions planned by step 9 and collect data at each
  way point,
    - save the data in directory `measurements`,
    - compute the pose of the camera in the end effector, and
    - save it in `config/calibrated-params.yaml`.

12. You can then commit the latter file and re-install agimus-demos package.
