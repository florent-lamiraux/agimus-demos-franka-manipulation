# Multiview data acquisition
import os
import time
import cv2
import rospy, tf2_ros
import numpy as np

from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def quat2SE3(quaternion):
    r = R.from_quat(quaternion[3:7])

    rotation_matrix = np.array(r.as_matrix())
    transformation_matrix = np.zeros((4,4))
    transformation_matrix[:3,:3] = rotation_matrix
    transformation_matrix[:3,3] = quaternion[0:3]
    transformation_matrix[3,3] = 1

    return transformation_matrix, quaternion

def get_cam_pose():
    try:
        rospy.init_node("inference_on_camera_image", anonymous=True)
    except:
        print("Ros node already initialized")
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    found = False
    out_of_time = False
    timer_in = time.time()
    timer_out = timer_in + 30
    while not found and not out_of_time:
        if tfBuffer.can_transform('camera_color_optical_frame', 'world', rospy.Time()):
            msg = tfBuffer.lookup_transform('camera_color_optical_frame', 'world', rospy.Time())
            x = msg.transform.translation.x
            y = msg.transform.translation.y
            z = msg.transform.translation.z
            theta_x = msg.transform.rotation.x
            theta_y = msg.transform.rotation.y
            theta_z = msg.transform.rotation.z
            theta_w = msg.transform.rotation.w
            cam_pose = [x, y, z, theta_x, theta_y, theta_z, theta_w]
            found = True
            print("cam pose :", cam_pose)
        if time.time() >= timer_out:
            out_of_time = True
            print("Runing out of time.")
            cam_pose = None
        time.sleep(1)
        print(int(time.time()-timer_in))

    return cam_pose

def capture_camera_image(name_img=None):
    dir_path = os.getcwd() + '/multiview/multiview_tless_2/images'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print("[INFO] Folder created")

    print("capture_camera_image_and_pose is running on realsense ros. Please assure that 'roslaunch realsense2_camera rs_camera.launch' is running.")
    
    # Initializing ROS node
    try:
        rospy.init_node("inference_on_camera_image", anonymous=True)
        print("[INFO] Ros node initialized.")
    except:
        print("Error initializing the ros node.")
        print("You can still capture camera shot.")
    
    # Capturing Video stream through ROS
    try:
        if name_img == None:
            name_img = str(dir_path+"image_" + str(time.time()) + ".png")
            for k in range(10):
                if not os.path.exists(str(dir_path+"/"+'0'+str(k+1)+'.png')):
                    name_img = str('0'+str(k+1)+'.png')
                    break
                else:
                    print("test not ok for index ",k)
        name = str(dir_path+"/"+name_img)
        print("Waiting to capture image from camera stream")
        image_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        print("[INFO] Saving the image ...")
        cv2.imwrite(name, image)
        print("Image captured. Saved under %s as %s." %(dir_path,name_img))
        return image, name_img
    except:
        print("Error in the for loop")
        return None, None

if __name__ == '__main__':
    print("[Start] data_acquisition tools")
