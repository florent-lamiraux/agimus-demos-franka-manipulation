# Instructions to run the demo

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
   roslaunch demo.launch
   ```

7. Start hppcorbaserver
   ```
   hppcorbaserver
   ```

8. Start the GUI
   ```
   gepetto-gui
   ```

9. launch the script that computes path starting from the initial configuration
   of the robot
   ```
   python - script_hpp.py
   ```

10. To execute a path
   ```
   rosrun agimus rqt_path_execution
   ```
