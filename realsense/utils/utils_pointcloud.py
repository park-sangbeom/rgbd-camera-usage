#!/usr/bin/env python
#-*- coding:utf-8 -*-
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from tf.transformations import rotation_matrix
import math
import numpy as np
import pcl
import tf
from utils.utils_pcl import * 
import cv_bridge 
import cv2 



def save_depth_img(msg_depth, name):
    bridge = cv_bridge.CvBridge()
    img_shape = (640, 480)
    try:
        cv_image_array = bridge.imgmsg_to_cv2(msg_depth, "32FC1")
        cv_image_array = np.array(cv_image_array, dtype = np.dtype('f8'))
        np.save("./data/npy/{}.npy".format(name), cv_image_array)
        cv2.imwrite('./data/png/{}.png'.format(name), cv_image_array*1)

        cv_image_array = cv2.resize(cv_image_array, img_shape, interpolation = cv2.INTER_CUBIC)
        cv_image_array = cv2.normalize(cv_image_array, cv_image_array, 0, 255, cv2.NORM_MINMAX)
        np.save("./data/npy/{}_norm.npy".format(name), cv_image_array)
        cv2.imwrite('./data/png/{}_norm.png'.format(name), cv_image_array*1)
        print("Max Depth: {}".format(np.max(cv_image_array))) 
        print("Min Depth: {}".format(np.min(cv_image_array)))
        print("Average Depth: {}".format(np.average(cv_image_array)))
        print('SAVED IMAGE')
        # cv2.imshow("Image from my node", cv_image_array*255)
        # cv2.waitKey(0)
    except cv_bridge.CvBridgeError as e:
        print(e)

# list를 pcd data로 바꾸는 함수
def change_list_to_pcd(lista):
    cloud = pcl.PointCloud_PointXYZRGB()
    cloud.from_list(lista)
    return cloud

# object 여러개를 리스트로 나누기
def get_obj_point(cluster_indices,white_cloud):
    obj_points = []
    for j,indices in enumerate(cluster_indices):
        point_list = []
        for k,indice in enumerate(indices):
            point_list.append([
                white_cloud[indice][0],
                white_cloud[indice][1],
                white_cloud[indice][2],
                1.5
            ])
        obj_points.append(point_list)
    return obj_points

# pcd data를 받아 중심점을 return하는 함수
def get_middle_point(pcd):
    x_total = 0
    y_total = 0
    z_total = []
    pcd_numpy = pcd.to_array()
    for i in range(len(pcd_numpy)):
        x_total += pcd_numpy[i,0]
        y_total += pcd_numpy[i,1]
        z_total.append(pcd_numpy[i,2])
    x = x_total / len(pcd_numpy)
    y = y_total / len(pcd_numpy)
    z = (max(z_total)+(min(z_total)))/2
    return x,y,z

# 복셀화(Down sampling)
def do_voxel_grid_downssampling(pcl_data,leaf_size):
    vox = pcl_data.make_voxel_grid_filter()
    vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
    return  vox.filter()

# 노이즈 제거
def do_statistical_outlier_filtering(pcl_data,mean_k,tresh):
    outlier_filter = pcl_data.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(mean_k)
    outlier_filter.set_std_dev_mul_thresh(tresh)
    return outlier_filter.filter()

# 바닥 제거 함수(random sample consensus 이용)
def do_ransac_plane_segmentation(pcl_data,pcl_sac_model_plane,pcl_sac_ransac,max_distance):
    '''
    Create the segmentation object
    :param pcl_data: point could data subscriber
    :param pcl_sac_model_plane: use to determine plane models, pcl.SACMODEL_PLANE
    :param pcl_sac_ransac: RANdom SAmple Consensus, pcl.SAC_RANSAC
    :param max_distance: Max distance for apoint to be considered fitting the model, 0.01
    :return: segmentation object
    '''
    seg = pcl_data.make_segmenter()
    seg.set_model_type(pcl_sac_model_plane)
    seg.set_method_type(pcl_sac_ransac)
    seg.set_distance_threshold(max_distance)

    # outliner 추출
    inliners, _ = seg.segment()
    inliner_object = pcl_data.extract(inliners,negative=False)
    outliner_object = pcl_data.extract(inliners,negative=True)
    return outliner_object

# clustering 함수
def euclid_cluster(cloud):
    white_cloud = XYZRGB_to_XYZ(cloud)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.015) # 0.015
    ec.set_MinClusterSize(300) # 20
    ec.set_MaxClusterSize(3000) # 3000
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    return cluster_indices, white_cloud

def cluster_mask(cluster_indices, white_cloud):
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([
                                            white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float( cluster_color[j] )
                                           ])
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    return cluster_cloud

# 관심 영역 설정
def do_passthrough(pcl_data,filter_axis,axis_min,axis_max):
    '''
    Create a PassThrough  object and assigns a filter axis and range.
    :param pcl_data: point could data subscriber
    :param filter_axis: filter axis
    :param axis_min: Minimum  axis to the passthrough filter object
    :param axis_max: Maximum axis to the passthrough filter object
    :return: passthrough on point cloud
    '''
    passthrough = pcl_data.make_passthrough_filter()
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    return passthrough.filter()

# 좌표 변환 함수
# def tf_matrix():
#     listener = tf.TransformListener() 
#     listener.waitForTransform('camera_link','camera_depth_optical_frame',rospy.Time(),rospy.Duration(2))
#     (t,q) = listener.lookupTransform('camera_link','camera_depth_optical_frame', rospy.Time(0))
#     t_matrix = tf.transformations.translation_matrix(t)
#     r_matrix = tf.transformations.quaternion_matrix(q)
#     return np.dot(t_matrix,r_matrix)

def change_frame(matt, points):
    transpose = points[:,0:3]
    ones = np.ones((len(points),1))
    transpose = np.concatenate((transpose,ones),axis=1)
    transpose = transpose.T
    transpose_after = np.dot(matt,transpose)
    transpose_after = transpose_after.T
    transpose_after_after = transpose_after[:,0:3]
    rgb = points[:,3].reshape(len(points),1)
    finalmat = np.concatenate((transpose_after_after,rgb),axis=1)
    return finalmat

def camera_to_base(matt, points):
    ones = np.ones((len(points),1))
    transpose = np.concatenate((points,ones),axis=1)
    transpose = transpose.T
    transpose_after = np.dot(matt,transpose)
    transpose_after = transpose_after.T
    transpose_after_after = transpose_after[:,0:3]
    return transpose_after_after

# numpy에서 pcd data로 바꾸는 함수
def numpy_to_pcd(nump):
    nump = nump.astype(np.float32)
    pcd = pcl.PointCloud_PointXYZI()
    pcd.from_array(nump)    
    return pcd

# callback 함수
def callback(input_ros_msg):
    cloud = ros_to_pcl(input_ros_msg)
    cloud = do_voxel_grid_downssampling(cloud,0.005)
    delete_floor = do_ransac_plane_segmentation(cloud,pcl.SACMODEL_PLANE,pcl.SAC_RANSAC,0.043)
    delete_desk = do_ransac_plane_segmentation(delete_floor,pcl.SACMODEL_PLANE,pcl.SAC_RANSAC,0.043)
    delete_desk = do_passthrough(delete_desk,'x',-0.4,0.4)
    delete_desk = do_passthrough(delete_desk,'y',-0.2, 1)
    # cloud = do_passthrough(cloud,'y',-0.2, 1)
    delete_desk_1 = delete_desk.to_array()
    # A = tf_matrix()
    # delete_desk_points = change_frame(A,delete_desk_1)
    # delete_desk_2 = numpy_to_pcd(delete_desk_points)
    delete_desk_2 = numpy_to_pcd(delete_desk_1)
    cluster_indices, white_cloud = euclid_cluster(delete_desk_2)
    obj_points = get_obj_point(cluster_indices,white_cloud)
    middle_point_lists = []
    for i in range(len(obj_points)):
        obj_group_cloud = change_list_to_pcd(obj_points[i])
        x,y,z = get_middle_point(obj_group_cloud)
        middle_point = [x,y,z]
        middle_point_lists.append(middle_point)
    get_color_list.color_list = []
    final = cluster_mask(cluster_indices,white_cloud)
    final_new = pcl_to_ros2(final)
    """ 
    NEED CAMERA CALIBRATIONA
    """
    rotation_mt_y = Rotation_Y(0.73)
    position_mt= Translation(0.115, 0 ,1.48)
    transform_mt = HT_matrix(rotation_mt_y,position_mt)
    middle_point_array = np.array(middle_point_lists)
    middle_points = list(middle_point_array)
    # change = camera_to_base(transform_mt,middle_point_array)
    # middle_points = list(change)
    # print(middle_points)
    # pub.publish(final_new)
    return middle_points

def rot_e():
    e = np.array([[1, 	       0, 	      0],
             	  [0,          1,         0],
             	  [0,          0,         1]])
    return e


def rot_x(rad):
    roll = np.array([[1, 	       0, 	         0],
             		 [0, np.cos(rad), -np.sin(rad)],
             		 [0, np.sin(rad),  np.cos(rad)]])
    return roll 


def rot_y(rad):
    pitch = np.array([[np.cos(rad), 0, np.sin(rad)],
                      [0,		    1, 	         0],
                      [-np.sin(rad),0, np.cos(rad)]])
    return pitch


def rot_z(rad):
    yaw = np.array([[np.cos(rad), -np.sin(rad),  0],
         	        [np.sin(rad),  np.cos(rad),  0],
              		[0, 			         0,  1]])
    return yaw 


def Rotation_E(): 
    e = np.array([[1, 	       0, 	      0,    0],
             	  [0,          1,         0,    0],
             	  [0,          0,         1,    0],
             	  [0,		   0,	      0,    0]])
    return e


def Rotation_X(rad):
    roll = np.array([[1, 	       0, 	      0,    0],
             		 [0, np.cos(rad), -np.sin(rad), 0],
             		 [0, np.sin(rad),  np.cos(rad), 0],
             		 [0,		   0,	      0,    0]])
    return roll 


def Rotation_Y(rad):
    pitch = np.array([[np.cos(rad), 0, np.sin(rad), 0],
              		  [0,		    1, 	         0, 0],
              		  [-np.sin(rad),0, np.cos(rad), 0],
              		  [0, 		    0, 	         0, 0]])
    return pitch


def Rotation_Z(rad):
    yaw = np.array([[np.cos(rad), -np.sin(rad),  0, 0],
         	        [np.sin(rad),  np.cos(rad),  0, 0],
              		[0, 			         0,  1, 0],
             		[0, 			         0,  0, 0]])
    return yaw 

def Translation(x , y, z):
    Position = np.array([[0, 0, 0, x],
                         [0, 0, 0, y],
                         [0, 0, 0, z],
                         [0, 0, 0, 1]])
    return Position


def HT_matrix(Rotation, Position):
    Homogeneous_Transform = Rotation + Position
    return Homogeneous_Transform


def pr2t(position, rotation): 
    position_4diag  = np.array([[0, 0, 0, position[0]],
                                [0, 0, 0, position[1]],
                                [0, 0, 0, position[2]], 
                                [0, 0, 0, 1]], dtype=object)
    rotation_4diag  = np.append(rotation,[[0],[0],[0]], axis=1)
    rotation_4diag_ = np.append(rotation_4diag, [[0, 0, 0, 1]], axis=0)
    ht_matrix = position_4diag + rotation_4diag_ 
    return ht_matrix


def t2p(ht_matrix):
    return ht_matrix[:-1, -1]


def t2r(ht_matrix):
    return ht_matrix[:-1, :-1]


def make_rotation(rad=0):
    for idx, rad_num in enumerate(rad.split()):
        if idx == 0 and float(rad_num) !=0:
            idx0 = rot_x(float(rad_num))
        elif idx==0 and float(rad_num) == 0: 
            idx0 = rot_e()
        if idx == 1 and float(rad_num) !=0:
            idx1 = rot_y(float(rad_num))
        elif idx==1 and float(rad_num) == 0: 
            idx1 = rot_e()
        if idx == 2 and float(rad_num) !=0:
            idx2 = rot_z(float(rad_num))
        elif idx==2 and float(rad_num)==0: 
            idx2 = rot_e()
    rot = idx2.dot(idx1).dot(idx0) 
    return rot

if __name__ == "__main__":
    rospy.init_node('pointcloud', anonymous=True)
    rospy.Subscriber('/camera/depth/color/points', PointCloud2, callback)
    pub = rospy.Publisher("/camera/depth/color/points_new",PointCloud2,queue_size=1)
    rospy.spin()