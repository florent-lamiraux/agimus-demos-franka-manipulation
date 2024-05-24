import sys
import time
import numpy as np
import tf2_ros, rospy
import tf2_geometry_msgs

from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from pyquaternion import Quaternion

class Object_Pose:
    def init(self, x, y, z, theta_x, theta_y, theta_z, theta_w):
        self.x = x
        self.y = y
        self.z = z
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.theta_z = theta_z
        self.theta_w = theta_w
        self.quaternion = [theta_x, theta_y, theta_z, theta_w]

    def __str__(self):
        return f"Poses : \n x = {self.x}\n y = {self.y}\n z = {self.z}\n theta x = {self.theta_x}\n theta y = {self.theta_y}\n theta z = {self.theta_z}\n theta w = {self.theta_w}\n"

    def isNormalize(self, to_normalize = False):
        normalized = False

        d = np.sqrt(self.theta_x**2 + self.theta_y**2 + self.theta_z**2 + self.theta_w**2)
        norm = self.theta_x/d + self.theta_y/d + self.theta_z/d + self.theta_w/d

        if norm >=0.99 and norm <= 1.01:
            normalized = True
        if normalized:
            print("The object quaternion is normalzed (norm =", norm,")")
        if not normalized:
            print("The object quaternion is not normalzed(norm =", norm,")")
        if to_normalize:
            quaternion = Quaternion(x=self.theta_x, y=self.theta_y, z=self.theta_z, w=self.theta_w)
            quaternion = Quaternion.normalised
            self.theta_x = quaternion[0]
            self.theta_y = quaternion[1]
            self.theta_z = quaternion[2]
            self.theta_w = quaternion[3]
            self.quaternion = quaternion

    def show_quaternion(self):
        print("Object quaternion :",self.quaternion)


def callback(msg):
    global message
    message = msg
    rospy.sleep(1)

def listen_to_happypose_detections():
    topic_name = "/happypose/detections"
    show_data = True
    break_var = False
    time_start = time.time()
    timeout = time_start + 10

    global object_poses
    global message

    while not break_var:
        if time.time() > timeout:
            break_var = True
            print("Elapsed time : 10s.")
        if message != None:
            break_var = True
            print("[INFO] Topic subscribed and objects found.")
        else :
            data = rospy.Subscriber(topic_name, Detection2DArray, callback)
            rospy.sleep(1)
            time_elapsed = time.time()-time_start
            sys.stdout.write("Current time : %d \r" % (time_elapsed))
            sys.stdout.flush()
        
        # Seperate the selected object from other detected object in the topic message
        select_objects('tless-obj_000001')

        # Transform object in world frame
        get_transform_all()
    
    if show_data:
        print("\n")
        print("<-----------------------POSES----------------------->")
        for el in object_poses:
            print(el)
            print(object_poses[el])
        print("<--------------------------------------------------->")

def select_objects(obj_name='tless-obj_000001'):
    global message
    global object_poses
    global object_message_list

    object_message_list = []
    nb_obj = len(message.detections)
    obj_id = 0

    for i in range(nb_obj):
        data = message.detections[i].results[0]
        name = data.hypothesis.class_id

        if obj_name in name:
            object_message_list.append(message.detections[i])
            obj_id += 1
            name = str(name+"_"+str(obj_id))
            x = data.pose.pose.position.x
            y = data.pose.pose.position.y
            z = data.pose.pose.position.z
            theta_x = data.pose.pose.orientation.x
            theta_y = data.pose.pose.orientation.y
            theta_z = data.pose.pose.orientation.z
            theta_w = data.pose.pose.orientation.w

            object_pose = Object_Pose()
            object_pose.x = x
            object_pose.y = y
            object_pose.z = z
            object_pose.theta_x = theta_x
            object_pose.theta_y = theta_y
            object_pose.theta_z = theta_z
            object_pose.theta_w = theta_w

            object_poses[str(name)] = object_pose 

def get_transform(obj_id=0):
    global object_message_list
    global tfBuffer
    global listener

    # Get the poses of the object in the world frame through tf_transform
    transform = tfBuffer.lookup_transform('world','camera_color_optical_frame', rospy.Time())
    pose_transformed = tf2_geometry_msgs.do_transform_pose(object_message_list[obj_id].results[0].pose, transform)

    return pose_transformed.pose

def get_transform_all():
    global object_message_list
    global tfBuffer
    global listener

    pose_list = []

    print(len(object_message_list))

    # Get the poses of all the objects in the world frame through tf_transform
    for obj_id in range(len(object_message_list)):
        transformed_pose = get_transform(obj_id)
        pose_list.append(transformed_pose)
    return pose_list

def run_pipeline():
    # Global variable
    global object_poses
    global message
    global object_message_list
    global tfBuffer
    global listener

    # Defining variable
    object_poses = {}

    # Create variable for data message
    message = None
    object_message_list = []

    # Initialize Ros node
    try:
        rospy.init_node("subscribe_to_miya_ros", anonymous=True)
        print("[INFO] Node Initialiazed ...")
    except:
        print("[INFO] Node already initialized ...")

    # Buffer and Subscriber
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    listen_to_happypose_detections()
    poses = get_transform_all()

    return poses

if __name__ == "__main__":
    print("[START]")

    # Global variable
    object_poses = {}
    name = None

    # Create global variable for data message
    message = None
    object_message_list = []

    # Initialize Ros node
    rospy.init_node("subscribe_to_miya_ros", anonymous=True)

    # Buffer and Subscriber
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
