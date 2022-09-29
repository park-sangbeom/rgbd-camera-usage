import sys 
sys.path.append("..")
import cv2 
from get_rostopic import ZED
import numpy as np 
import rospy 
import matplotlib.pyplot as plt 
from utils.utils_pcl import * 
from utils.utils_pointcloud import * 


if __name__=="__main__":
    rospy.init_node('ZED')
    mode     = "depth"
    camera   = ZED(mode=mode)
    np.save("./data/npy/ros.npy", camera.depth_image)
    save_depth_img(camera.depth_image)