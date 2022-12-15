#-*-coding:utf-8-*-
"""
    Utility functions
    @author: Qianyue He
    @date: 2022-12-13
"""

import scipy.io
import cv2 as cv
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

__T = np.float32([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
])

def get_base_path(case_idx: int = 1):
    return f"../diff_1/raw_data/case{case_idx}"

def get_path(case_idx: int = 1, img_idx: int = 1, no_scat = False):
    return f"{get_base_path(case_idx)}/{'no_' if no_scat else ''}scat/img_{'no' if no_scat else ''}haze{img_idx}.png"

def imshow(name: str, pic: np.ndarray):
    print("Press any key to quit.")
    while True:
        cv.imshow(name, pic)
        key = cv.waitKey(30)
        if key > 0:
            cv.destroyAllWindows()
            return key == 27
        
def skew_symmetric_transform(t: np.ndarray):
    x, y, z = t
    return np.float32([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])
    
def visualize_equalized_hist(case_idx = 1, img_idx = 3, disp = False):
    img_path = get_path(case_idx, img_idx)
    img = cv.imread(img_path)
    img_eq = np.stack([cv.equalizeHist(img[..., i]) for i in range(3)], axis = -1)
    if disp:
        imshow("equalized hist", img_eq)
    return img_eq

def coarse_matching(c_img: np.ndarray, o_img: np.ndarray, raw_kpts_cp: np.ndarray, raw_kpts_op: np.ndarray):
    kpts_cp = [cv.KeyPoint(*pt, 1) for pt in raw_kpts_cp]
    kpts_op = [cv.KeyPoint(*pt, 1) for pt in raw_kpts_op]

    extractor = cv.SIFT.create(32)
    (kpts_cp, feats_cp) = extractor.compute(c_img, kpts_cp)
    (kpts_op, feats_op) = extractor.compute(o_img, kpts_op)
    macther = cv.FlannBasedMatcher()
    matches = macther.match(feats_cp, feats_op)
    return kpts_cp, feats_cp, kpts_op, feats_op, matches

def get_features(case_idx: int = 1, pic_id: int = 1, center_id: int = 3):
    mat_file_path = f"{get_base_path(case_idx)}/keypoints.mat"
    feature_mat = scipy.io.loadmat(mat_file_path)["keypoints"]
    valid_pic_id = {1, 2, 3, 4, 5}
    valid_pic_id.remove(center_id)
    if not pic_id in valid_pic_id:
        raise ValueError(f"{pic_id} not valid (3 is the id of the center image, therefore [1, 2, 4, 5] are available)")

    feat_mat = feature_mat[pic_id - 1 if pic_id < center_id else pic_id - 2][0]
    raw_kpts_op = feat_mat[3:-1, :].T              # raw keypoints of non-center pictures, discarding the homogeneous dim
    raw_kpts_cp = feat_mat[:2, :].T                # raw keypoints of the center pictures, discarding the homogeneous dim
    return raw_kpts_cp, raw_kpts_op

"""
    Inverse projection via intrinsic matrix
    Transform pixel coordinates (non-homogeneous) to world frame ray direction    
    Logic should be checked (matplotlib plot3d)
    pix: shape (N, 2, 1) N = number of covisible frames
"""
def world_frame_ray_dir(K_inv: np.ndarray, pix: np.ndarray, Rs: np.ndarray):
    num_covi = pix.shape[0]
    homo_pix = np.concatenate((pix, np.ones((num_covi, 1, 1))), axis = 1)      # shape (N, 3, 1)
    camera_coords = K_inv @ homo_pix                                            # K_inv for inverse projection
    print("Camera: ", camera_coords.ravel())
    init_w_coords = __T @ camera_coords
    print("Init: ", init_w_coords.ravel())
    init_w_coords = init_w_coords / np.linalg.norm(init_w_coords, axis = 1)[:, None, :]     # normalize along axis 0
    return Rs @ init_w_coords                # normalized ray direction in the world frame

def read_rot_trans_int(path: str):
    poses = pd.read_excel(path, sheet_name="Parameters of UAV").to_numpy()[..., 1:].astype(np.float32)
    position = poses[..., :3]
    euler_angles = poses[..., 3:]
    Rs = []
    for euler in euler_angles:
        # Roll(y axis), Yaw (z axis), Pitch (x axis)
        euler[1] = -euler[1]
        r: Rot = Rot.from_euler("yzx", euler, degrees = True)          # deg 2 rad
        Rs.append(r.as_matrix())
    R = np.stack(Rs, axis = 0)
    K = pd.read_excel(path, sheet_name="Parameters of camera").to_numpy()[..., 1:].astype(np.float32)

    # shape R: (N, 3, 3), position: (N, 3), K: (3, 3)
    return R, position, K
