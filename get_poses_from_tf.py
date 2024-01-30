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
    global list_name
    global x
    global y
    global z
    global theta_x
    global theta_y
    global theta_z
    global theta_w

    name = msg.transforms[0].child_frame_id
    if 'tless' in name and name not in list_name:
        print("Object found :", name)
        list_name.append(msg.transforms[0].child_frame_id)

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
    rospy.Subscriber("tf", TFMessage, callback)
    # rospy.spin()

def setting_variables():
    print("Setting up the global variables.")
    object_poses = {}
    list_name = []
    x = None
    y = None
    z = None
    theta_x = None
    theta_y = None
    theta_z = None
    theta_w = None

    return object_poses, list_name, x,y,z,theta_x,theta_y,theta_z,theta_w

def get_poses(init_node=True):
    print("[INFO] Launch the happypose_inference service (30s before timeout).")
    
    break_var = False
    global object_poses
    global list_name
    global x
    global y
    global z
    global theta_x
    global theta_y
    global theta_z
    global theta_w

    # Checking if variables exists
    print("Checking if variables exists.")
    try:
        x not in globals()
    except:
        print("[INFO] variables not in globals")
        try:
            x not in locals()
        except:
            print("[INFO] variables not in locals")
            object_poses, list_name, x,y,z,theta_x,theta_y,theta_z,theta_w = setting_variables()

    # Resetting the global variables
    if x != None:
        print("Resetting the global variables.")
        object_poses = {}
        list_name = []
        x = None
        y = None
        z = None
        theta_x = None
        theta_y = None
        theta_z = None
        theta_w = None
    else:
        print("Variables set.")

    if init_node:
        print("Initializing ROS node")
        rospy.init_node('listener', anonymous=True)

    time_start = time.time()
    timeout = time_start + 30

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

    print("\n",list_name)
    print("\n",object_poses)
    print(len(object_poses))

    return object_poses,list_name


if __name__ == '__main__':
    print("[START]")
    print("get_poses_from_tf ready !")

    # Global variable
    list_name = []
    object_poses = {}
    x = None
    y = None
    z = None
    theta_x = None
    theta_y = None
    theta_z = None
    theta_w = None

    rospy.loginfo("")
