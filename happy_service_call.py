#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import rospy, tf, tf2_ros
import multiprocessing as mp

from tf2_msgs.msg import TFMessage
from ros_cosypose.srv import Inference, InferenceRequest, InferenceResponse

def get_number_of_objects():
    print("[INFO] Calling the service ...")
    rospy.wait_for_service("happypose_inference")
    try:
        collect_nb = rospy.ServiceProxy("happypose_inference", Inference)
        response = collect_nb()
    except rospy.ServiceException as e:
        print("Service call failed : %s"%e)
        response = None
    return response

def convert_std_msg_to_array(std_msg):
    string = str(std_msg)
    id = string.find('[')
    string = string[id:len(string)]
    print("splited string :",string)
    converted_array = eval(string)
    print("Number of object in index 1 : ",converted_array[1])
    print("Number of object in index 2 : ",converted_array[2])
    return converted_array

def callback(msg):
    global name
    global x
    global y
    global z
    global theta_x
    global theta_y
    global theta_z
    global theta_w

    name = msg.transforms[0].child_frame_id
    x = msg.transforms[0].transform.translation.x
    y = msg.transforms[0].transform.translation.y
    z = msg.transforms[0].transform.translation.z
    theta_x = msg.transforms[0].transform.rotation.x
    theta_y = msg.transforms[0].transform.rotation.y
    theta_z = msg.transforms[0].transform.rotation.z
    theta_w = msg.transforms[0].transform.rotation.w

def listener():
    # rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("tf", TFMessage, callback)
    # rospy.spin()

def pose_getter():
    list_pose = {}
    x = None
    process = mp.Process(target=rospy.ServiceProxy("happypose_inference", Inference))

    print("[INFO] Starting the inference process")
    # process.start()

    while x == None:
        listener()

    print("Pose of the object ",name,": ")
    print("x : ",x)
    print("y : ",y)
    print("z : ",z)
    print("theta x : ",theta_x)
    print("theta y : ",theta_y)
    print("theta z : ",theta_z)
    print("theta w : ",theta_w)
                
    #Tests fait dans le script get_poses_from_tf_test.py dans /home/dorian/Downloads.
    
    return list_pose


def service_call(Tless_1_and_2 = False):
    print("[START] Starting the happypose service call.")
    call = get_number_of_objects()
    nb_obj = 0
    if call != None:
        list_obj = convert_std_msg_to_array(call)
        if Tless_1_and_2:
            print("[INFO] Counting object_tless_1 and object_tless_2 ...")
            nb_obj = list_obj[1] + list_obj[2]
        else:
            print("[INFO] Counting all the object ...")
            for el in list_obj:
                nb_obj += el
        print("Number of objects :",nb_obj)
    else:
        print("No object detected or the service call failed.")
    return nb_obj

if __name__ == '__main__':
    rospy.init_node('service')  
    list_pose = service_call()
