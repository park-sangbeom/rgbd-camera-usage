import rospy 
from sensor_msgs.msg import PointCloud2, Image
import time 
import numpy as np 
import message_filters

class ZED():
    def __init__(self, mode="pointcloud"):    
        self.tick = 0
        self.mode = mode 
        self.point_cloud = None 
        self.depth_image = None 
        self.rgb_image = None 
        self.pointcloud_pub = rospy.Publisher("/point_cloud_new", PointCloud2, queue_size=1)
        self.rgb_image_sub = message_filters.Subscriber('/zed2i/zed_node/rgb_raw/image_raw_color', Image)
        if mode=="pointcloud":
            self.point_cloud_sub = message_filters.Subscriber('/zed2i/zed_node/point_cloud/cloud_registered', PointCloud2)
            self.ts = message_filters.TimeSynchronizer([self.point_cloud_sub, self.rgb_image_sub], 10)
        else: 
            self.depth_image_sub = message_filters.Subscriber('/zed2i/zed_node/depth/depth_registered', Image)
            self.ts = message_filters.TimeSynchronizer([self.depth_image_sub, self.rgb_image_sub], 10)

        self.ts.registerCallback(self.callback)
        
        tic_temp = 0
        while self.tick<2:
            time.sleep(1e-3)
            tic_temp = tic_temp + 1
            if tic_temp > 5000:
                print ("[ERROR] CHECK REALSENSE435")
                break

    def callback(self, depth_msg, rgb_msg):
        self.tick = self.tick+1
        self.color_image = rgb_msg
        if self.mode == "pointcloud":
            self.point_cloud = depth_msg
        else: 
            self.depth_image = depth_msg 
        

if __name__=="__main__":
    mode = "pointcloud"
    rospy.init_node('ZED')
    camera=ZED(mode=mode)
    depth_data = camera.point_cloud
    image_data = camera.color_image
    np.save("zed1_depth.npy",depth_data)
    np.save("zed1_image.npy", image_data)