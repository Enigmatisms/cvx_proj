#-*-coding:utf-8-*-
"""
    Spectral weighting with pair-wise consistency estimation
    @author: Qianyue He
    @date: 2022-12-14
"""

__all__ = ['spectral_method']

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from numpy import ndarray as Arr
from model import LMSSolver
from options import get_options

CENTER_PIC_ID = 3

def get_coarse_matches(center_img: Arr, other_img: Arr, case_idx: int = 1, pic_id: int = 1):
    raw_kpts_cp, raw_kpts_op = get_features(case_idx, pic_id, CENTER_PIC_ID)
    # return --- kpts_cp, feats_cp, kpts_op, feats_op, matches
    return coarse_matching(center_img, other_img, raw_kpts_cp, raw_kpts_op)

def match_RANSAC(kpts_cp: Arr, kpts_op: Arr, matches: Arr, swap = False):
    if swap:
        dst_pts, src_pts = cv_to_array(kpts_cp, kpts_op, matches)
    else:
        src_pts, dst_pts = cv_to_array(kpts_cp, kpts_op, matches)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    mask = mask.astype(np.float32)
    return H, mask.ravel()

def calculate_M(
    kpts_cp: list, feats_cp: Arr, 
    kpts_op: list, feats_op: Arr, 
    F: Arr, matches: Arr, 
    f_scaler: float = 0.5, eps: float = 30.0, threshold: float = 0.6, 
    verbose = False, swap = True, output_ransac = True
):
    """ 
        Calculate adjacency matrix M: diagonal part: mathcing score, off-diagonal part: consistency score
        - kpts_cp: keypoints for center image, feats_cp: features for center image
        - kpts_op: keypoints for image in other view, feats_op: features for image in other view
        - F: fundamental matrix for epipolar constraint score
        - matches: coarse matches produced by FLANN, f_scaler: score scaler for epipolar constraints
        - threshold: segmentation score bigger than this will be counted
        - swap: without swap (swap = False), center_image will be warpped to the other image 
        - For loop free, as SIMD as possible via numpy
    """
    if output_ransac:
        H, ransac_mask = match_RANSAC(kpts_cp, kpts_op, matches, swap)
    else:
        H           = None
        ransac_mask = None
    num_matches = len(matches)
    M = np.zeros((num_matches, num_matches))
    
    # computing diagonal part
    src_pts, dst_pts = cv_to_array(kpts_cp, kpts_op, matches)
    homo_src_pts     = np.hstack((src_pts, np.ones((num_matches, 1))))     # homogeneous coordinates
    homo_dst_pts     = np.hstack((dst_pts, np.ones((num_matches, 1))))     # homogeneous coordinates
    c_feats, o_feats = cv_to_array(feats_cp, feats_op, matches, is_pts = False)
    
    c_feats    /= np.linalg.norm(c_feats, axis = -1, keepdims=True)
    o_feats    /= np.linalg.norm(o_feats, axis = -1, keepdims=True)
    epi_vectors = F @ homo_src_pts.T                 # src_pts is of shape (N, 3), transpose it -> (3, N)
    epi_score   = np.abs(np.sum(homo_dst_pts * epi_vectors.T, axis = -1))
    match_score = np.sum(c_feats * o_feats, axis = -1)
    M           = np.diag(match_score + f_scaler / (1. + epi_score))
    # computing off-diagonal part
    rcp_value   = 1 / 2 / (eps ** 2)
    src_matrix  = np.sum((src_pts.reshape(-1, 1, 2) - src_pts.reshape(1, -1, 2)) ** 2, axis = -1)    # (N, N), distance between points in src img
    dst_matrix  = np.sum((dst_pts.reshape(-1, 1, 2) - dst_pts.reshape(1, -1, 2)) ** 2, axis = -1)    # (N, N), distance between points in dst img
    dist_matrix = ((src_matrix - dst_matrix) ** 2) * rcp_value
    off_score   = np.maximum(4.5 - dist_matrix, 0.)         # clamp
    np.fill_diagonal(off_score, 0.)                         # diagonal is already computed
    M          += off_score                                 # segmentation matrix is fully computed

    U, _, _  = np.linalg.svd(M)         # SVD decomposition for the principle eigenvector
    segment  = np.abs(U[:, 0])
    segment /= np.max(segment)
    segment[segment < 1e-6] = 0
    if verbose:
        for i, m_value in enumerate(ransac_mask):
            valid = (m_value > 0) if output_ransac else "Unknown"
            print(f"Matching score: {M[i, i]:.4f}\tvalid match: {valid}/{segment[i]}")
        plt.imshow(M)
        plt.colorbar()
        plt.show()
    bool_mask = segment > threshold
    ransac_mask *= threshold
    ransac_mask[bool_mask] = segment[bool_mask]     # array thresholding
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
    
def model_solve(kpts_cp: list, kpts_op: list, matches: list, weights: Arr, param = 0.5, verbose = False, swap = True) -> Arr:
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
    
    solver = LMSSolver(param)
    return solver.solve(pts_c, pts_o, selected_weights, verbose = verbose, swap = swap)

# Packaged function for multi-threading / easier calling
def spectral_method(opts):
    # TODO: better image denoising model
    center_img  = visualize_equalized_hist(case_idx = opts.case_idx, img_idx = CENTER_PIC_ID)
    other_img   = visualize_equalized_hist(case_idx = opts.case_idx, img_idx = opts.img_idx)
    
    kpts_cp, feats_cp, kpts_op, feats_op, matches \
        = get_coarse_matches(center_img, other_img, opts.case_idx, opts.img_idx)
    F   = get_fundamental(opts.case_idx, CENTER_PIC_ID, opts.img_idx)
    
    weights, ransac_mask, H = calculate_M(
        kpts_cp, feats_cp, kpts_op, feats_op, F, matches, 
        threshold = opts.threshold, eps = opts.affinity_eps, f_scaler = opts.epi_weight, verbose = False
    )           # To visualize the result of Affinity Matrix calculation, make verbose True
    
    if opts.viz == 'ransac':
        visualize_weighted(center_img, other_img, kpts_cp, kpts_op, matches, ransac_mask)           # visualize RANSAC mask
    elif opts.viz == 'spectral':
        visualize_weighted(center_img, other_img, kpts_cp, kpts_op, matches, weights)               # visualize spectral score

    H_pred = model_solve(kpts_cp, kpts_op, matches, ransac_mask, param = opts.huber_param, verbose = opts.verbose)

    if opts.verbose:
        print("Ground truth homography: ", H.ravel())
    
    if opts.save_warpped:
        center_img_nc, other_img_nc = get_no_scat_img(opts.case_idx, opts.img_idx, CENTER_PIC_ID)
        warpped_baseline = image_warping(center_img_nc, other_img_nc, H, False)
        warpped_result   = image_warping(center_img_nc, other_img_nc, H_pred, False)
        cv.imwrite(f"./output/lms.png", warpped_result)
        cv.imwrite("./output/baseline.png", warpped_baseline)
        
if __name__ == "__main__":
    opts = get_options()
    spectral_method(opts)
    