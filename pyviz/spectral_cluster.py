#-*-coding:utf-8-*-
"""
    Python visualization of SIFT-RANSAC baseline blendering (Ghost Effect) 
    This is the baseline model
    @author: Qianyue He
    @date: 2022-12-12
"""

import cv2 as cv
import numpy as np
from sys import argv
from utils import *
import matplotlib.pyplot as plt
from numpy import ndarray as Arr
from sdp_problem import SDPSolver
from itertools import compress

CENTER_PIC_ID = 3

def get_coarse_matches(center_img: Arr, other_img: Arr, case_idx: int = 1, pic_id: int = 1):
    raw_kpts_cp, raw_kpts_op = get_features(case_idx, pic_id, CENTER_PIC_ID)
    # return --- kpts_cp, feats_cp, kpts_op, feats_op, matches
    return coarse_matching(center_img, other_img, raw_kpts_cp, raw_kpts_op)

def match_RANSAC(kpts_cp: Arr, kpts_op: Arr, matches: Arr):
    src_pts = np.float32([ kpts_cp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpts_op[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    mask = mask.astype(np.float32)
    return H, mask

# get fundamental matrix through given pose and camera intrinsic
def fundamental(Rc, Ro, tc, to, K):
    Ro_inv = np.linalg.inv(Ro)
    Rr = Ro_inv @ Rc                        # relative rotation (transform point in center frame to other frame)
    tr = Ro_inv @ (tc - to)                 # relative translation
    ss_t = skew_symmetric_transform(tr)     # 3 * 3 skew symmetric matrix
    K_inv = np.linalg.inv(K)
    return K_inv.T @ ss_t @ Rr @ K_inv

# Get matrix M, F is fundamental matrix
def calculate_M(
    kpts_cp: list, feats_cp: Arr, 
    kpts_op: list, feats_op: Arr, 
    F: Arr, matches: Arr, 
    f_scaler: float = 0.5, eps: float = 30.0, 
    threshold: float = 0.6, verbose = False, output_ransac = True
):
    if output_ransac:
        H, ransac_mask = match_RANSAC(kpts_cp, kpts_op, matches)
    else:
        H           = None
        ransac_mask = None
    num_matches = len(matches)
    M = np.zeros((num_matches, num_matches))
    # computing diagonal part
    for i, match in enumerate(matches):
        c_feat = feats_cp[match.queryIdx]
        o_feat = feats_op[match.trainIdx]
        c_feat /= np.linalg.norm(c_feat)
        o_feat /= np.linalg.norm(o_feat)
        
        xc, yc = kpts_cp[match.queryIdx].pt
        xo, yo = kpts_op[match.trainIdx].pt
        c_pix  = np.float32([[xc], [yc], [1]])
        o_pix  = np.float32([[xo, yo, 1]])
        epi_score = abs(o_pix @ F @ c_pix)
        M[i, i] = np.sum(c_feat * o_feat) + f_scaler / (1. + epi_score)
    # computing off-diagonal part
    rcp_value = 1 / 2 / (eps ** 2)
    for i in range(num_matches):
        for j in range(i, num_matches):
            if i == j: continue
            m1, m2 = matches[i], matches[j]
            xc_1, yc_1 = kpts_cp[m1.queryIdx].pt
            xo_1, yo_1 = kpts_op[m1.trainIdx].pt
            
            xc_2, yc_2 = kpts_cp[m2.queryIdx].pt
            xo_2, yo_2 = kpts_op[m2.trainIdx].pt
            d1 = (xc_1 - xc_2) ** 2 + (yc_1 - yc_2) ** 2
            d2 = (xo_1 - xo_2) ** 2 + (yo_1 - yo_2) ** 2
            score = max(0, 4.5 - ((d1 - d2) ** 2) * rcp_value)
            M[i, j] = score
            M[j, i] = score
    U, _, _  = np.linalg.svd(M)
    segment  = np.abs(U[:, 0])
    segment /= np.max(segment)
    segment[segment < 1e-6] = 0
    if verbose:
        for i, (match, m_value) in enumerate(zip(matches, ransac_mask)):
            valid = (m_value > 0) if output_ransac else "Unknown"
            print(f"Matching score: {M[i, i]:.4f}\tvalid match: {valid}/{segment[i]}")
        plt.imshow(M)
        plt.colorbar()
        plt.show()
    for i in range(num_matches):
        w = segment[i]
        if w > threshold:
            ransac_mask[i] = w
        else:
            ransac_mask[i] *= threshold
    return segment, ransac_mask, H
    
def visualize_weighted(c_img: Arr, o_img: Arr, kpts_cp: list, kpts_op: list, matches: list, weight: Arr):
    offset_x = c_img.shape[1]
    out_image = np.concatenate((c_img, o_img), axis = 1)
    for m, w in zip(matches, weight):
        if w <= 1e-3: continue
        xc, yc = kpts_cp[m.queryIdx].pt
        xo, yo = kpts_op[m.trainIdx].pt
        p1 = np.int32([xc, yc])
        p2 = np.int32([xo + offset_x, yo])
        cv.line(out_image, p1, p2, (0, int(255 * w), 0), 1)
        cv.circle(out_image, p1, 5, (0, int(255 * w), 0), 1)
        cv.circle(out_image, p2, 5, (0, int(255 * w), 0), 1)
    imshow("weighted", out_image)
    
def sdp_model_solve(kpts_cp: list, kpts_op: list, matches: list, weights: Arr) -> Arr:
    weights = weights.ravel()
    pts_c = []
    pts_o = []
    selected_weights = []
    for m, w in zip(matches, weights):
        if w <= 1e-3: continue
        xc, yc = kpts_cp[m.queryIdx].pt
        xo, yo = kpts_op[m.trainIdx].pt
        pts_c.append([xc, yc])
        pts_o.append([xo, yo])
        selected_weights.append(w)
        
    pts_c = np.float32(pts_c)
    pts_o = np.float32(pts_o)
    selected_weights = np.float32(selected_weights)
    
    solver = SDPSolver(0.5, 0.5)
    return solver.solve(pts_c, pts_o, selected_weights, verbose = True)
        
if __name__ == "__main__":
    argv_len = len(argv)
    case_idx = 1
    img_idx = 1
    if argv_len > 1:
        case_idx = int(argv[1]) 
    if argv_len > 2:
        img_idx = int(argv[2]) 
        
    center_img = visualize_equalized_hist(case_idx = case_idx, img_idx = CENTER_PIC_ID)
    other_img = visualize_equalized_hist(case_idx = case_idx, img_idx = img_idx)
    Rs, ts, K = read_rot_trans_int(f"{get_base_path(case_idx)}/Parameters.xlsx")
    print(K)
    kpts_cp, feats_cp, kpts_op, feats_op, matches = get_coarse_matches(center_img, other_img, case_idx, img_idx)
    Rc = Rs[CENTER_PIC_ID - 1]
    tc = ts[CENTER_PIC_ID - 1]
    Ro = Rs[img_idx - 1]
    to = ts[img_idx - 1]
    F = fundamental(Rc, Ro, tc, to, K)
    weights, ransac_mask, H = calculate_M(kpts_cp, feats_cp, kpts_op, feats_op, F, matches)
    visualize_weighted(center_img, other_img, kpts_cp, kpts_op, matches, weights)
    # visualize_weighted(center_img, other_img, kpts_cp, kpts_op, matches, ransac_mask)
    H_sdp = sdp_model_solve(kpts_cp, kpts_op, matches, ransac_mask)
    print("Ground truth homography: ", H.ravel())
    
    center_img_nc, other_img_nc = get_no_scat_img(case_idx, img_idx, CENTER_PIC_ID)
    warpped_baseline = image_warping(other_img_nc, center_img_nc, H, False)
    warpped_sdp      = image_warping(other_img_nc, center_img_nc, H_sdp, False)
    cv.imwrite("./output/sdp.png", warpped_sdp)
    cv.imwrite("./output/baseline.png", warpped_baseline)