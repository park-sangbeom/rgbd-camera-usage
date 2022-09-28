import cv2 
from get_rostopic import ZED
import numpy as np 
import rospy 
import cv_bridge 
from utils.utils_pcl import * 
from utils.utils_pointcloud import *

if __name__=="__main__":
    rospy.init_node('ZED')
    mode = "pointcloud"
    camera=ZED(mode=mode)
    pcl_data = ros_to_pcl(camera.point_cloud)
    print(type(pcl_data))
    print(pcl_data.to_array())
    # points = ros_to_numpy(camera.point_cloud)
