#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import rospy
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

def service_call():
    print("[START] Starting the happypose service call.")
    call = get_number_of_objects()
    nb_obj = 0
    if call != None:
        list_obj = convert_std_msg_to_array(call)
        for el in list_obj:
            nb_obj += el
        print("Number of objects :",nb_obj)
    else:
        print("No object detected or the service call failed.")
    return nb_obj

if __name__ == '__main__':
    service_call()