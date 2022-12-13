"""
    Python test for SfM using the given feature points
    If this module succeeds in doing the things I want, it can be scaled into our model
    Co-visibility SfM feature point position optimization
    @author: Qianyue He
    @date: 2022-12-13
"""

import scipy
import scipy.io
import cv2 as cv
import numpy as np
import pandas as pd
from utils import get_base_path, visualize_equalized_hist, imshow
from scipy.spatial.transform import Rotation as Rot
from itertools import compress

CENTER_PIC_ID = 3
PIC_LIST = (1, 2, 3, 4, 5)

def has_affinity(res_src_pts: list, pt: np.ndarray) -> int:
    for i, kpt in enumerate(res_src_pts):
        diff = np.abs((kpt - pt))
        if (diff < 1e-1).all():     # close enough
            return i
    return -1

# Read Rotation, Translation and Intrinsic Matrix
def read_rot_trans_int(path: str):
    poses = pd.read_excel(path, sheet_name="Parameters of UAV").to_numpy()[..., 1:].astype(np.float32)
    position = poses[..., :3]
    euler_angles = poses[..., 3:]
    Rs = []
    for euler in euler_angles:
        # Roll(y axis), Yaw (z axis), Pitch (x axis)
        r: Rot = Rot.from_euler("yzx", euler, degrees = True)          # deg 2 rad
        Rs.append(r.as_matrix())
    R = np.stack(Rs, axis = 0)
    K = pd.read_excel(path, sheet_name="Parameters of camera").to_numpy()[..., 1:].astype(np.float32)

    # shape R: (N, 3, 3), position: (N, 3), K: (3, 3)
    return R, position, K

def match_all(case_idx: int, debug_disp: bool = False, verbose: bool = False):
    # Data preparation
    mat_file_path = f"{get_base_path(case_idx)}/keypoints.mat"
    feature_mat = scipy.io.loadmat(mat_file_path)["keypoints"]
    imgs = [visualize_equalized_hist(case_idx = case_idx, img_idx = idx) for idx in PIC_LIST]
    # Rs, ps, K = read_rot_trans_int(f"{get_base_path(case_idx)}/Parameters.xlsx")
    center_img = imgs[CENTER_PIC_ID - 1]
    
    all_src_pts = []
    all_dst_pts = []
    all_matches = []
    
    for idx in PIC_LIST:
        if idx == CENTER_PIC_ID: continue       # skip the center image
        other_img = imgs[idx - 1]
        
        feat_mat = feature_mat[idx - 1 if idx < CENTER_PIC_ID else idx - 2][0]
        raw_kpts_op = feat_mat[3:-1, :].T              # raw keypoints of non-center pictures, discarding the homogeneous dim
        raw_kpts_cp = feat_mat[:2, :].T                # raw keypoints of the center pictures, discarding the homogeneous dim

        kpts_cp = [cv.KeyPoint(*pt, 1) for pt in raw_kpts_cp]
        kpts_op = [cv.KeyPoint(*pt, 1) for pt in raw_kpts_op]
        extractor = cv.SIFT.create(32)
        (kpts_cp, feats_cp) = extractor.compute(center_img, kpts_cp)
        (kpts_op, feats_op) = extractor.compute(other_img, kpts_op)
        macther = cv.FlannBasedMatcher()
        matches = macther.match(feats_cp, feats_op)
        src_pts = np.float32([ kpts_cp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpts_op[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        _, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        
        valid_matches = list(compress(matches, mask))           # get valid matches using mask
        valid_src_pts = list(compress(src_pts, mask))
        all_src_pts.append(valid_src_pts)
        all_matches.append(valid_matches)
        all_dst_pts.append(dst_pts)

        if debug_disp:
            out_image = np.concatenate((center_img, other_img), axis = 1)
            draw_params = dict(matchColor = (0, 255, 0),
                                singlePointColor = None,
                                matchesMask = mask.ravel().tolist(),
                                flags = 2
            )

            out_img = cv.drawMatches(center_img, kpts_cp, other_img, kpts_op, matches, out_image, **draw_params)
            imshow(f"match-{idx}-{CENTER_PIC_ID}", out_img)

        if verbose:
            print(f"Matching image {idx} against image {CENTER_PIC_ID}...")
            print(f"Number of valid matches: {mask.sum()}. Number of total matches: {len(matches)}")
    return imgs, all_src_pts, all_dst_pts, all_matches

"""
    get COVIsibility Dictionary
"""
def get_covid(all_src_pts: list, all_matches: list):
    all_len = len(all_src_pts)
    res_src_pts: list = all_src_pts[0]                                      # list of KeyPoint
    covid = [[match.trainIdx, -1, -1, -1] for match in all_matches[0]]      # Interesting naming: covisibility dictionary (src->dst)
    for target_id in range(1, all_len):
        src_pts = all_src_pts[target_id]
        input_matches = all_matches[target_id]
        for kpt, match in zip(src_pts, input_matches):
            aff = has_affinity(res_src_pts, kpt)
            if aff >= 0:
                covid[aff][target_id] = match.trainIdx                      # kpt exists in res_src_pts, no need to append
            else:
                new_match = [-1 for _ in range(4)]
                new_match[target_id] = match.trainIdx
                covid.append(new_match)
                res_src_pts.append(kpt)                                     # A new kpt
    return res_src_pts, covid

def visualize_covid(all_imgs: list, covid: list, res_src_pts: list, all_dst_pts: list, verbose = False):
    multi_covid_cnt = 0
    for i, item in enumerate(covid): 
        if (np.int32(item) >= 0).sum() <= 1: continue
        if verbose: print(item)
        cx, cy = res_src_pts[i].ravel()
        center_img = all_imgs[CENTER_PIC_ID - 1].copy()
        center_img = cv.circle(center_img, (int(cx), int(cy)), 3, (0, 255, 0), -1)
        imgs_disp = [center_img]
        for j, idx in enumerate(item):
            if idx == -1: continue
            img_idx = j if j < CENTER_PIC_ID - 1 else j + 1
            print(img_idx)
            px, py = all_dst_pts[j][idx].ravel()
            out_img = all_imgs[img_idx].copy()
            out_img = cv.circle(out_img, (int(px), int(py)), 3, (0, 255, 0), -1)
            imgs_disp.append(out_img)
        result_img = np.concatenate(imgs_disp, axis = 1)
        if imshow("COVID", result_img):
            print("Early break.")
            break
        multi_covid_cnt += 1
    if verbose:
        print(f"Item with covisibility: {multi_covid_cnt} out of {len(covid)}")
    

if __name__ == "__main__":
    # path = f"{get_base_path(1)}/Parameters.xlsx"
    # read_rot_trans_int(path)
    imgs, all_src_pts, all_dst_pts, all_matches = match_all(1, False, False)
    res_src_pts, covid = get_covid(all_src_pts, all_matches)
    visualize_covid(imgs, covid, res_src_pts, all_dst_pts, True)