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

__all__ = [                                                                        
        'get_base_path', 'read_rot_trans_int', 'get_no_scat_img', 'get_features',    
        'save2mat', 'imshow', 'visualize_equalized_hist', 'image_warping',            
        'cv_to_array', 'coarse_matching', 'get_fundamental', 'world_frame_ray_dir',
        'normalized_feature'
]

# ============================== IO Utilities ==================================
def get_base_path(case_idx: int = 1):
    return f"../diff_1/raw_data/case{case_idx}"

def get_path(case_idx: int = 1, img_idx: int = 1, no_scat = False):
    return f"{get_base_path(case_idx)}/{'no_' if no_scat else ''}scat/img_{'no' if no_scat else ''}haze{img_idx}.png"

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

def get_no_scat_img(case_idx, img_idx, center_id):
    img_path = get_path(case_idx, img_idx, True)
    ctr_path = get_path(case_idx, center_id, True)
    return cv.imread(ctr_path), cv.imread(img_path)

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

def save2mat(path: str, arr: np.ndarray, name: str = 'sift_feature', prefix: str = './output/'):
    mat_file_path = f"{prefix}{path}.mat"
    scipy.io.savemat(mat_file_path, {name: arr})

# ============================== IO Utilities ==================================


# ============================== Visualization utilities =================================
def imshow(name: str, pic: np.ndarray):
    print("Press any key to quit.")
    while True:
        cv.imshow(name, pic)
        key = cv.waitKey(30)
        if key > 0:
            cv.destroyAllWindows()
            return key == 27

def visualize_equalized_hist(case_idx = 1, img_idx = 3, disp = False):
    img_path = get_path(case_idx, img_idx)
    img = cv.imread(img_path)
    img_eq = np.stack([cv.equalizeHist(img[..., i]) for i in range(3)], axis = -1)
    if disp:
        imshow("equalized hist", img_eq)
    return img_eq

def image_warping(img_base: np.ndarray, img2warp: np.ndarray, H: np.ndarray, direct_blend = True):
    """
        Warp image given homography matrix    
        - direct blend: if True --- warpped pixel will cover the base image, 
        - otherwise mean will be computed (ghost effect can be observed) 
    """
    h1,w1 = img_base.shape[:2]
    h2,w2 = img2warp.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv.perspectiveTransform(pts2, H)                            # warpped image boundaries
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis = 0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis = 0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([
        [1, 0, t[0]],
        [0, 1, t[1]],
        [0, 0, 1]]
    ) # translate

    result = cv.warpPerspective(img2warp, Ht.dot(H), (xmax - xmin, ymax - ymin))
    if direct_blend == False:           # mean-blendering, this is much slower. Yet we can observe the ghost effect in here
        offset_x, offset_y = t
        for row_id in range(offset_y, h1 + offset_y):
            y_base = row_id - offset_y
            for col_id in range(offset_x, w1 + offset_x):
                x_base = col_id - offset_x
                if any(result[row_id, col_id]):     # non-empty: mean blendering
                    result[row_id, col_id] = ((img_base[y_base, x_base].astype(np.float32) + result[row_id, col_id].astype(np.float32)) / 2.).astype(np.uint8)
                else:                               # empty: superposition
                    result[row_id, col_id] = img_base[y_base, x_base]
    else:
        result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img_base
    return result

# ============================== Visualization utilities =================================


# ============================ Transform utilities =============================
def cv_to_array(source: list, target: list, matches: list, is_pts = True):
    if is_pts:
        src = np.float32([ source[m.queryIdx].pt for m in matches ])
        dst = np.float32([ target[m.trainIdx].pt for m in matches ])
    else:
        src = np.stack([ source[m.queryIdx] for m in matches ], axis = 0)
        dst = np.stack([ target[m.trainIdx] for m in matches ], axis = 0)
    return src, dst

def coarse_matching(c_img: np.ndarray, o_img: np.ndarray, raw_kpts_cp: np.ndarray, raw_kpts_op: np.ndarray):
    kpts_cp = [cv.KeyPoint(*pt, 1) for pt in raw_kpts_cp]
    kpts_op = [cv.KeyPoint(*pt, 1) for pt in raw_kpts_op]

    extractor = cv.SIFT.create(nfeatures = 128)
    (kpts_cp, feats_cp) = extractor.compute(c_img, kpts_cp)
    (kpts_op, feats_op) = extractor.compute(o_img, kpts_op)
    macther = cv.FlannBasedMatcher()
    matches = macther.match(feats_cp, feats_op)
    return kpts_cp, feats_cp, kpts_op, feats_op, matches

def normalized_feature(feats_cp: np.ndarray, feats_op: np.ndarray, match: cv.DMatch):
    feat_c = feats_cp[match.queryIdx]
    feat_o = feats_op[match.trainIdx]
    return feat_c / np.linalg.norm(feat_c), feat_o / np.linalg.norm(feat_o)

# ============================ Transform utilities =============================


# ================================ Mathematical utilities ====================================
def skew_symmetric_transform(t: np.ndarray):
    x, y, z = t
    return np.float32([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

# get fundamental matrix through given pose and camera intrinsic
def fundamental(Rc, Ro, tc, to, K):
    Ro_inv = np.linalg.inv(Ro)
    Rr = Ro_inv @ Rc                        # relative rotation (transform point in center frame to other frame)
    tr = Ro_inv @ (tc - to)                 # relative translation
    ss_t = skew_symmetric_transform(tr)     # 3 * 3 skew symmetric matrix
    K_inv = np.linalg.inv(K)
    return K_inv.T @ ss_t @ Rr @ K_inv

# Get fundamental matrix
def get_fundamental(case_idx: int, center_idx: int, img_idx: int):
    Rs, ts, K = read_rot_trans_int(f"{get_base_path(case_idx)}/Parameters.xlsx")
    Rc = Rs[center_idx - 1]
    tc = ts[center_idx - 1]
    Ro = Rs[img_idx - 1]
    to = ts[img_idx - 1]
    return fundamental(Rc, Ro, tc, to, K)

def world_frame_ray_dir(K_inv: np.ndarray, pix: np.ndarray, Rs: np.ndarray):
    """
        Inverse projection via intrinsic matrix
        - Transform pixel coordinates (non-homogeneous) to world frame ray direction    
        - Logic should be checked (matplotlib plot3d)
        - pix of shape (N, 2, 1) N = number of covisible frames
    """
    num_covi = pix.shape[0]
    homo_pix = np.concatenate((pix, np.ones((num_covi, 1, 1))), axis = 1)      # shape (N, 3, 1)
    camera_coords = K_inv @ homo_pix                                            # K_inv for inverse projection
    print("Camera: ", camera_coords.ravel())
    init_w_coords = __T @ camera_coords
    print("Init: ", init_w_coords.ravel())
    init_w_coords = init_w_coords / np.linalg.norm(init_w_coords, axis = 1)[:, None, :]     # normalize along axis 0
    return Rs @ init_w_coords                # normalized ray direction in the world frame

# ================================ Mathematical utilities ====================================

if __name__ == "__main__":
    visualize_equalized_hist(1, 1, True)