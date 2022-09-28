import sys 
sys.path.append("..")
import cv2 
from get_rostopic import RealsenseD435i
import numpy as np 
import rospy 
import matplotlib.pyplot as plt 
from utils.utils_pcl import ros_to_pcl, ros_to_numpy, XYZRGB_to_XYZ
from utils.utils_pointcloud import * 
print("Done.")

if __name__ =="__main__":
    rospy.init_node('Realsense')
    mode     = "depth"
    camera   = RealsenseD435i(mode=mode)
    np.save("test.npy", camera.depth_image)
    save_depth_img(camera.depth_image)