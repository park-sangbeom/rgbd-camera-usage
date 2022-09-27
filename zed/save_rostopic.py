import cv2 
from get_rostopic import ZED
import numpy as np 
import rospy 
import cv_bridge 
from utils.pcl_helper import * 

bridge = cv_bridge.CvBridge()


def save_depth_img(msg_depth):
    img_shape = (640, 360)
    try:
        cv_image_array = bridge.imgmsg_to_cv2(msg_depth, "32FC1")
        cv_image_array = np.array(cv_image_array, dtype = np.dtype('f8'))
        cv_image_array = cv2.resize(cv_image_array, img_shape, interpolation = cv2.INTER_CUBIC)
        cv_image_array = cv2.normalize(cv_image_array, cv_image_array, 0, 255, cv2.NORM_MINMAX)
        print(cv_image_array.shape)
        print(type(cv_image_array))
        print(cv_image_array[0:10,0:10])
        print("Max Depth: {}".format(np.max(cv_image_array))) 
        print("Min Depth: {}".format(np.min(cv_image_array)))
        print("Average Depth: {}".format(np.average(cv_image_array)))
        cv2.imwrite('depth_test2.png', cv_image_array*255)
        print('SAVED IMAGE')
        # cv2.imshow("Image from my node", cv_image_array*255)
        # cv2.waitKey(0)

    except cv_bridge.CvBridgeError as e:
        print(e)
    

if __name__=="__main__":
    rospy.init_node('ZED')
    mode = "pointcloud"
    camera=ZED(mode=mode)
    pcl_data = ros_to_pcl(camera.point_cloud)
    print(type(pcl_data))
    print(pcl_data.to_array())
    # points = ros_to_numpy(camera.point_cloud)
