import sys 
sys.path.append("..")
sys.path.append("realsense")

import cv2 
from get_rostopic import RealsenseD435i
import numpy as np 
import rospy 
import matplotlib.pyplot as plt 
from utils.utils_pcl import ros_to_pcl, ros_to_numpy, XYZRGB_to_XYZ
from utils.utils_pointcloud import * 
import time 

rospy.init_node('Realsense')
mode     = "pointcloud"
name     = "test"
camera   = RealsenseD435i(mode=mode)
save_pc(msg_pc=camera.point_cloud, name=name)