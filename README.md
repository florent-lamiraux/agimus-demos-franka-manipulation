# Troubleshoooting

If you encounter computer slowing down are freeze, restart the process hppcorbaserver.

If the /happypose/detections topic don't appear anymore in the rostopic list, use the command `docker compose restart` in the happypose_ros/panda_happypose_bridge folder.

# Instructions to run the demo

1. connect to the ip adress of the robot using firefox. On akasan, this
   address is 172.17.1.3.

2. release the brakes,
3. activate FCI,

4. in terminal, change de $ROS_MASTER_URI and $ROS_IP to 'ROS_MASTER_URI=http://140.93.16.70:11311' and 'ROS_IP=140.93.16.70'
   You can also enter for Akasan : 'export ROS_MASTER_URI=http://140.93.16.70:11311 export ROS_IP=140.93.16.70'

5. in terminal 1, in this directory
   ```bash
   cd ~/devel/src/agimus-demos/franka/manipulation
   roslaunch robot.launch arm_id:=panda2 robot_ip:=172.17.1.3
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

9. In the docker on Miyanoura, start those 2 commands :
   ```
   ros2 launch realsense2_camera rs_camera.py
   ros2 launch ros_bridge
   ```

<br/>

# Multiview
<details>
<summary>Multiview</summary>

10. In the docker, run the happypose multiview command : 
   ```bash
   ros2 launch happypose multiview
   ```

11. Start gathering the images.
   To do so move the robot do the desired position, where all objects are visible, and run this command :
   ```
   ros2 run multiview ... #
   ```
   Replace the # with the number of your shot (from 0 to a max of 9). Your last shot should be :
   ```
   ros2 run multiview ... 9
   ```

12. Publish cam view to cam topic for happypose multiview to process the images.
   ```
   ros2 topic ...
   ```

13. Happypose will publish the poses in the topic /happypose/detections.

14. in another terminal you can run ```python -i script_hpp.py``` and get the poses with ```btf.run_pipeline()```.
</details>

<br/>

# Singleview
<details>
<summary>Singleview</summary>
10. In the docker, run the happypose multiview command : 
   ```
   ros2 launch happypose multiview
   ```

11. Start gathering the images.
   To do so move the robot do the desired position, where all objects are visible, and run this command :
   ```
   ros2 run multiview ... #
   ```
   Replace the # with the number of your shot (from 0 to a max of 9). Your last shot should be :
   ```
   ros2 run multiview ... 9
   ```

12. Publish cam view to cam topic for happypose multiview to process the images.
   ```
   ros2 topic ...
   ```

13. Happypose will publish the poses in the topic /happypose/detections.

14. in another terminal you can run ```python -i script_hpp.py``` and get the poses with ```btf.run_pipeline()```.

</details>

<br/>

# Playing the generated path on the robot

To execute a path you have 2 options.

First one with graphic interface to select the path you want to play :
   ```
   rosrun agimus rqt_path_execution
   ```

The second one if you want to play the latest path added to the ProblemSolver (only available in script_hpp) :
   ```
   move_robot()
   ```

