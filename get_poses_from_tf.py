#!/usr/bin/env python

import sys
import rospy, tf
import time
import multiprocessing as mp
from std_msgs.msg import String
from geometry_msgs.msg import Pose, TransformStamped
from tf2_msgs.msg import TFMessage

def callback(msg):
    
    global object_poses
    global x
    global y
    global z
    global theta_x
    global theta_y
    global theta_z
    global theta_w

    nb_of_frame = len(msg.transforms)

    name = msg.transforms[0].child_frame_id
    if 'tless' in name and name not in List_name:
        print(name," : ", 0)
        List_name.append(msg.transforms[0].child_frame_id)

        x = msg.transforms[0].transform.translation.x
        y = msg.transforms[0].transform.translation.y
        z = msg.transforms[0].transform.translation.z
        theta_x = msg.transforms[0].transform.rotation.x
        theta_y = msg.transforms[0].transform.rotation.y
        theta_z = msg.transforms[0].transform.rotation.z
        theta_w = msg.transforms[0].transform.rotation.w

        object_poses[str(name)] = {}
        object_poses[str(name)]['x'] = x
        object_poses[str(name)]['y'] = y
        object_poses[str(name)]['z'] = z
        object_poses[str(name)]['theta x'] = theta_x
        object_poses[str(name)]['theta y'] = theta_y
        object_poses[str(name)]['theta z'] = theta_z
        object_poses[str(name)]['theta w'] = theta_w
        

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("tf", TFMessage, callback)
    # rospy.spin()

def main_function():
    print("[INFO] Launch the happypose_inference service (30s before timeout).")
    
    break_var = False
    timeout = time.time() + 30
    time_start = time.time()

    while x == None and not break_var:
        if time.time() > timeout:
            break_var = True
            print("Elapsed time : 30s.")
        listener()
        rospy.sleep(1)
        time_elapsed = time.time()-time_start
        sys.stdout.write("Current time : %d \r" % (time_elapsed))
        # sys.stdout.write("Current state of break : %d \r" % (break_var))
        sys.stdout.flush()

    print(List_name)
    print(x)

if __name__ == '__main__':
    print("[START]")
    print("get_poses_from_tf ready !")
    # Global variable
    List_name = []
    object_poses = {}
    x = None
    y = None
    z = None
    theta_x = None
    theta_y = None
    theta_z = None
    theta_w = None

    rospy.loginfo("")
