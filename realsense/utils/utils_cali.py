import apriltag
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

import apriltag
import math

def r2rpy(R,unit='rad'):
    """
        Rotation matrix to roll,pitch,yaw in radian
    """
    roll  = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], (math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    yaw   = math.atan2(R[1, 0], R[0, 0])
    if unit == 'rad':
        out = np.array([roll, pitch, yaw])
    elif unit == 'deg':
        out = np.array([roll, pitch, yaw])*180/np.pi
    else:
        out = None
        raise Exception("[r2rpy] Unknown unit:[%s]"%(unit))
    return out    

def get_apriltag_pose(img, img_depth, intrinsic_matrix, tag_size=0.008, verbose_bbox=False, verbose_pose=True):
    """
        In AX=XB Equation, (extrinsic calibration) 
        Get matrix about A that represents detected AprilTag pose in camera coordinate.
    """
    # camera parameter setting.
    fx = intrinsic_matrix.fx
    fy = intrinsic_matrix.fy
    ppx = intrinsic_matrix.ppx
    ppy = intrinsic_matrix.ppy

    cam_params = [fx, fy, ppx, ppy]

    # apriltag setting.
    detector = apriltag.Detector()

    img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_Gray = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
    img_xyz = compute_xyz(img_depth, camera_info=intrinsic_matrix)   # 3d estimated img

    results = detector.detect(img_Gray)

    if verbose_bbox:
        print(results)

    # Check the detections on the image
    if len(results) > 0:
        draw_bbox(results, img, intrinsic_matrix=intrinsic_matrix, verbose_bbox=False, plot_img=False)

        for r in results:
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))

            pose, e0, e1 = detector.detection_pose(detection=r, camera_params=cam_params, tag_size=tag_size)    # should check tag_size
            
            poseRotation = pose[:3, :3]
            poseTranslation = pose[:3, 3]
    
            center_point = [int(r.center[i]) for i in range(2)]    # in int type

            rot_april = pose[:3, :3]
            center_3d = np.array([img_xyz[center_point[1]][center_point[0]]])   # order of pixel array is y, x 

            T_april = np.concatenate((rot_april, center_3d.T), axis=1)  # 4x3 matrix
            T_april = np.concatenate((T_april, np.array([[0,0,0,1]])), axis=0)  # 4x4 matrix
        
            if verbose_pose:
                cv2.putText(img, f"[RPY]: {r2rpy(rot_april, unit='deg')[0]:.2f}, {r2rpy(rot_april, unit='deg')[1]:.2f}, {r2rpy(rot_april, unit='deg')[2]:.2f}", \
                            (0, 30), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 125), 2)
                cv2.putText(img, f"[x,y,z]: {center_3d[0][0]:.2f}, {center_3d[0][1]:.2f}, {center_3d[0][2]:.2f}", \
                            (0, 60), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 125), 2)
                # plt.imshow(img)
                # plt.show()

        return T_april, img

    else:   # if any detected marker is none, return None.
        print("There's no april tag")
        return None, img


def draw_bbox(results, image, intrinsic_matrix, verbose_bbox=False, plot_img=True):
    width = intrinsic_matrix.width
    height = intrinsic_matrix.height

    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (255, 0, 0), 3)
        cv2.line(image, ptB, ptC, (255, 0, 0), 3)
        cv2.line(image, ptC, ptD, (255, 0, 0), 3)
        cv2.line(image, ptD, ptA, (255, 0, 0), 3)

        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
    
        if verbose_bbox:
            cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 255, 255), 3)
            # print("Apriltag name: {}".format(tagFamily))

            x_centered = cX - width / 2
            y_centered = -1 * (cY - height / 2)

            cv2.putText(image, f"Center X coord: {x_centered}", (ptB[0] + 10, ptB[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (125, 0, 125), 7)

            cv2.putText(image, f"Center Y coord: {y_centered}", (ptB[0] + 10, ptB[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (125, 0, 125), 7)

            cv2.putText(image, f"Tag ID: {r.tag_id}", (ptC[0] - 70, ptC[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 0, 125), 7)

        # cv2.circle(image, (int((width / 2)), int((height / 2))), 5, (0, 0, 255), 2)

    if plot_img:
        plt.imshow(image)
        plt.show()

def convert_from_uvd(u, v, d, cam_matrix):
    """
        pxToMetre: Constant, depth scale factor
        cx: Center x of Camera
        cy: Center y of Camera
        focalx: Focal length
        focaly: Focal length 
    """

    pxToMetre = 1

    focalx = cam_matrix[0][0]
    cx = cam_matrix[0][2]
    focaly = cam_matrix[1][1]
    cy = cam_matrix[1][2]

    d *= pxToMetre
    x_over_z = (cx - u) / focalx
    y_over_z = (cy - v) / focaly
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z

    return -y, x, z

def compute_xyz(depth_img, camera_info):

    # , fx, fy, px, py, height, width
    fx = camera_info.fx
    cx = camera_info.ppx
    fy = camera_info.fx
    cy = camera_info.ppy

    height = camera_info.height
    width = camera_info.width

    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    
    z_e = depth_img
    x_e = (indices[..., 1] - cx) * z_e / fx
    y_e = (indices[..., 0] - cy) * z_e / fy
    
    # Order of y_ e is reversed !
    xyz_img = np.stack([-y_e, x_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img

import numpy as np
import scipy
# from util_fk import T2axisangle, skew, T2aa

def T2axisangle(T):
    """
        T to axis-angle representation.
    """   
    R = T[:3,:3]

    theta_axisangle = math.acos((np.trace(R)-1)/2)    # in rad unit.

    prefix_multiplier = 1 / (2*math.sin(theta_axisangle))

    rx = prefix_multiplier * (R[2][1] - R[1][2]) * theta_axisangle
    ry = prefix_multiplier * (R[0][2] - R[2][0]) * theta_axisangle
    rz = prefix_multiplier * (R[1][0] - R[0][1]) * theta_axisangle

    rot_axisangle = np.array([rx, ry, rz])

    return rot_axisangle, theta_axisangle

def skew(R):    # return ske
    """
        Convert to skew matrix.
    """   
    return np.array([[0, -R[2], R[1]],
                     [R[2], 0, -R[0]],
                     [-R[1], R[0], 0]])


def T2aa(T):
    T = np.array(T, dtype=np.float64, copy=False)

    rot = T[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, W = np.linalg.eig(rot.T)
    
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    
    axis = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R corresponding to eigenvalue of 1
    w, Q = np.linalg.eig(T)
    
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")

    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]

    # rotation angle depending on axis
    cosa = (np.trace(rot) - 1.0) / 2.0
    if abs(axis[2]) > 1e-8:
        sina = (T[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
    elif abs(axis[1]) > 1e-8:
        sina = (T[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
    else:
        sina = (T[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]

    angle = math.atan2(sina, cosa)
    return axis, angle, point


def log_group_skew(T):
    """
        Convert to Log matrix form. Return skew matrix form.
        This is code of Lemma 2.
    """
    R = T[0:3, 0:3] # Rotation matrix
    # Lemma 2
    theta = np.arccos((np.trace(R) - 1)/2)

    logr = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * theta / (2*np.sin(theta))
    return logr # return skew matrix form

def log_group_matrix(T):
    """
        Convert to Log matrix form. Return 3x3 matrix form.
        This is code of Lemma 2.
    """
    R = T[0:3, 0:3] # Rotation matrix
    # Lemma 2
    theta = np.arccos((np.trace(R) - 1)/2)

    theta_ = R - R.T
    log_r = theta_ * theta / (2*np.sin(theta))
    return log_r

def get_extrinsic_calibration_park(A, B):
    """
        Solve AX=XB calibration.
    """

    data_num = len(A)
    
    M = np.zeros((3,3))
    C = np.zeros((3*data_num, 3))
    d = np.zeros((3*data_num, 1))

    # columns of matrix A, B
    alpha = log_group_skew(A[0])     # 3 x 1
    beta = log_group_skew(B[0])      # 3 x 1
    alpha_2 = log_group_skew(A[1]) # 3 x 1
    beta_2 = log_group_skew(B[1])  # 3 x 1
    alpha_3 = np.cross(alpha, alpha_2)  # 3 x 1
    beta_3 = np.cross(beta, beta_2)     # 3 x 1

    # print(alpha)
    # print(alpha_2)
    # print(beta)
    # print(beta_2)
    # assert np.array_equal(np.cross(alpha, alpha_2), np.zeros(3)) 
    # assert np.array_equal(np.cross(beta, beta_2), np.zeros(3))

    # M = \Sigma (beta * alpha.T)
    M1 = np.dot(beta.reshape(3,1),alpha.reshape(3,1).T)
    M2 = np.dot(beta_2.reshape(3,1),alpha_2.reshape(3,1).T)
    M3 = np.dot(beta_3.reshape(3,1),alpha_3.reshape(3,1).T)
    M = M1+M2+M3

    # theta_x = (M.T * M)^(-1/2) * M.T    
    # theta_x = np.dot(np.sqrt(np.linalg.inv((np.dot(M.T, M)))), M.T)
    # RuntimeWarning: invalid value encountered in sqrt: np.sqrt results nan values
    theta_x = np.dot(scipy.linalg.sqrtm(np.linalg.inv((np.dot(M.T, M)))), M.T)  # rotational info

    # A_ = np.array([alpha, alpha_2, alpha_3])
    # B_ = np.array([beta, beta_2, beta_3])
    # B_inv = np.linalg.inv(B_)
    # theta_x = A * B_inv

    for i in range(data_num):
        A_rot   = A[i][0:3,0:3]
        A_trans = A[i][0:3, 3]
        B_rot   = B[i][0:3,0:3]
        B_trans = B[i][0:3, 3]
        
        C[3*i:3*i+3, :] = np.eye(3) - A_rot
        d[3*i:3*i+3, 0] = A_trans - np.dot(theta_x, B_trans)


    b_x = np.dot(np.linalg.inv(np.dot(C.T, C)), np.dot(C.T, d))     # translational info

    return theta_x, b_x


def estimate_translation(A, B, Rx):
    """
    Estimate the translation component of :math:`\hat{X}` in :math:`AX=XB`. This
    requires the estimation of the rotation component :math:`\hat{R}_x`
    Parameters
    ----------
    A: list
        List of homogeneous transformations with the relative motion of the
        end-effector
    B: list
        List of homogeneous transformations with the relative motion of the
        calibration pattern (often called `object`)
    Rx: array_like
        Estimate of the rotation component (rotation matrix) of :math:`\hat{X}`
    Returns
    -------
    tx: array_like
        The estimated translation component (XYZ value) of :math:`\hat{X}`
    """
    C = []
    d = []
    for Ai, Bi in zip(A, B):
        ta = Ai[:3, 3]
        tb = Bi[:3, 3]
        C.append(Ai[:3, :3]-np.eye(3))
        d.append(np.dot(Rx, tb)-ta)
    C = np.array(C)
    C.shape = (-1, 3)
    d = np.array(d).flatten()
    tx, residuals, rank, s = np.linalg.lstsq(C, d, rcond=-1)
    
    return tx.flatten()


def get_extrinsic_calibration_tsai(A, B):
    """
        Implementation of Tsai method Extrinsic calibration.
    """
    norm = np.linalg.norm
    C = []
    d = []

    for Ai, Bi in zip(A, B):
        # Transform the matrices to their axis-angle representation
        # r_gij, theta_gij = T2axisangle(Ai)
        # r_cij, theta_cij = T2axisangle(Bi)

        r_gij, theta_gij, _ = T2aa(Ai)
        r_cij, theta_cij, _ = T2aa(Bi)

        # Tsai uses a modified version of the angle-axis representation
        Pgij = 2*np.sin(theta_gij/2.)*r_gij
        Pcij = 2*np.sin(theta_cij/2.)*r_cij

        # Use C and d to avoid overlapping with the input A-B
        C.append(skew(Pgij+Pcij))
        d.append(Pcij-Pgij)

    # Estimate Rx
    C = np.array(C)
    C.shape = (-1, 3)

    d = np.array(d).flatten()

    Pcg_, residuals, rank, s = np.linalg.lstsq(C, d, rcond=-1)
    Pcg = 2*Pcg_ / np.sqrt(1 + norm(Pcg_)**2)

    R1 = (1 - norm(Pcg)**2/2.) * np.eye(3)
    R2 = (np.dot(Pcg.reshape(3, 1), Pcg.reshape(1, 3)) +
            np.sqrt(4-norm(Pcg)**2) * skew(Pcg)) / 2.
    Rx = R1 + R2

    # Estimate tx
    tx = estimate_translation(A, B, Rx)
    
    # Return X
    X = np.eye(4)
    X[:3, :3] = Rx
    X[:3, 3] = tx
    return X

#### Method 2: Angle-axis representation. (Frank. Park, 1994)
def get_extrinsic_calibration_frank(A, B):
    M = np.zeros((3, 3))
    for Ai, Bi in zip(A, B):
        # Transform the matrices to their axis-angle representation
        axis, angle, _ = T2aa(Ai)
        alpha = angle*axis

        axis, angle, _ = T2aa(Bi)
        beta = angle*axis

        # Compute M
        M += np.dot(beta.reshape(3, 1), alpha.reshape(1, 3))
        
    # Estimate Rx
    Rx = np.dot(np.linalg.inv(scipy.linalg.sqrtm(np.dot(M.T, M))), M.T)
    # Estimate tx
    tx = estimate_translation(A, B, Rx)

    # Return T
    T = np.eye(4)
    T[:3, :3] = Rx
    T[:3, 3] = tx
    
    return T