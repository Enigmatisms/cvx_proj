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
from model import SDPSolver, LMSSolver
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

def recompute_matching(
    kpts_cp: list, feats_cp: Arr, 
    kpts_op: list, feats_op: Arr, 
    matches: list, H: Arr, opts,
) -> Arr:
    """
        Using merely opencv RANSAC may lead to degraded result: 
        - SIFT descriptor not similar enough but warped points are actually close
        - Therefore, after computing the global Homography, we should recompute the matches
        - and start all over. Kind of feels like E-M algorithm
        
        args:
        - kpts_cp, kpts_op: keypoints on center / the other image
        - feats_cp, feats_op: SIFT features on center / the other image
        - matches: coarse matches
        - H: estimated global homography matrix
        - radius: two points, if the dist between each other after warping is within radius, then valid match
        - score_thresh: SIFT features should be similar enough
    """
    mask = np.zeros(len(matches), dtype = np.float32)
    for i, m in enumerate(matches):
        feat_c, feat_o = normalized_feature(feats_cp, feats_op, m)
        kpt_c = np.float32(kpts_cp[m.queryIdx].pt)
        kpt_o = np.matmul(H, np.float32((*kpts_op[m.trainIdx].pt, 1)))       # make homogeneous -> warp -> inhomogeneous
        kpt_o = (kpt_o / kpt_o[2])[:-1]
        dist = np.linalg.norm(kpt_o - kpt_c)
        feat_score = np.sum(feat_c * feat_o)
        if dist < opts.em_radius and feat_score > opts.score_thresh:
            mask[i] = 1.
    return mask

def calculate_M(
    kpts_cp: list, feats_cp: Arr, 
    kpts_op: list, feats_op: Arr, 
    F: Arr, matches: Arr, opts, 
    verbose = False, swap = True, init_ransac = True, Hg = None
):
    """ 
        Calculate adjacency matrix M: diagonal part: mathcing score, off-diagonal part: consistency score
        - kpts_cp: keypoints for center image, feats_cp: features for center image
        - kpts_op: keypoints for image in other view, feats_op: features for image in other view
        - F: fundamental matrix for epipolar constraint score
        - matches: coarse matches produced by FLANN, f_scaler: score scaler for epipolar constraints
        - threshold: segmentation score bigger than this will be counted
        - swap: without swap (swap = False), center_image will be warpped to the other image 
        - Hg: global homography, if not None, use this to estimate matching mask
        - For loop free, as SIMD as possible via numpy
    """
    if init_ransac:
        if Hg is not None:
            # re-estimate matchings use Hg here
            H = Hg
            ransac_mask = recompute_matching(kpts_cp, feats_cp, kpts_op, feats_op, matches, Hg, opts)
        else:
            H, ransac_mask  = match_RANSAC(kpts_cp, kpts_op, matches, swap)
        original_mask   = ransac_mask.copy()
    else:
        H               = None
        ransac_mask     = None
        original_mask   = None
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
    M           = np.diag(match_score + opts.epi_weight / (1. + epi_score))
    # computing off-diagonal part
    rcp_value   = 1 / 2 / (opts.affinity_eps ** 2)
    src_matrix  = np.sum((src_pts.reshape(-1, 1, 2) - src_pts.reshape(1, -1, 2)) ** 2, axis = -1)    # (N, N), distance between points in src img
    dst_matrix  = np.sum((dst_pts.reshape(-1, 1, 2) - dst_pts.reshape(1, -1, 2)) ** 2, axis = -1)    # (N, N), distance between points in dst img
    dist_matrix = ((src_matrix - dst_matrix) ** 2) * rcp_value
    off_score   = np.maximum(4.5 - dist_matrix, 0.)         # clamp
    np.fill_diagonal(off_score, 0.)                         # diagonal is already computed
    M          += off_score                                 # segmentation matrix is fully computed

    segment  = np.abs(U[:, 0])
    segment /= np.max(segment)
    segment[segment < 1e-6] = 0
    if verbose:
        for i, m_value in enumerate(ransac_mask):
            valid = (m_value > 0) if init_ransac else "Unknown"
            print(f"Matching score: {M[i, i]:.4f}\tvalid match: {valid}/{segment[i]}")
        plt.imshow(M)
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    bool_mask = segment > opts.aff_thresh
    ransac_mask *= opts.aff_thresh
    ransac_mask[bool_mask] = segment[bool_mask]     # array thresholding
    return segment, H, ransac_mask, original_mask
    
def visualize_weighted(
    c_img: Arr, o_img: Arr, 
    kpts_cp: list, kpts_op: list, 
    matches: list, weight: Arr, 
    mask_prev = None, draw_zero_w = False, disp = False, only_diff = False
):
    offset_x = c_img.shape[1]
    out_image = np.concatenate((c_img, o_img), axis = 1)
    for i, (m, w) in enumerate(zip(matches, weight)):
        minor_weight = w <= 1e-3
        if minor_weight and not draw_zero_w: continue
            
        xc, yc = kpts_cp[m.queryIdx].pt
        xo, yo = kpts_op[m.trainIdx].pt
        p1 = np.int32([xc, yc])
        p2 = np.int32([xo + offset_x, yo])
        color = (0, 0, 255) if minor_weight else (0, int(255 * w), 0)
        if mask_prev is not None:
            if (mask_prev[i] < 1e-4) ^ minor_weight:        # XOR logic: minor_weight = False (True) & mask_prev[i] = 1 & 0
                color = (0, 255, 255)
            elif only_diff:
                continue
        if not minor_weight:
            cv.line(out_image, p1, p2, color, 2)
        cv.circle(out_image, p1, 4, color, 2)
        cv.circle(out_image, p2, 4, color, 2)
    if disp:
        imshow("weighted", out_image)
    return out_image
    
def model_solve(kpts_cp: list, kpts_op: list, matches: list, weights: Arr, param = 0.5, max_iter = 8000, verbose = False, swap = True, lms = True) -> Arr:
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
    
    if lms:
        solver = LMSSolver(max_iter, param)
    else:   
        solver = SDPSolver(max_iter, param, param)
    return solver.solve(pts_c, pts_o, selected_weights, verbose = verbose, swap = swap)

# Packaged function for multi-threading / easier calling
def spectral_method(opts, img_idx = None):
    # TODO: better image denoising model
    H_pred      = None
    mask_prev   = None
    image_idx   = opts.img_idx if img_idx is None else img_idx
    center_img  = visualize_equalized_hist(case_idx = opts.case_idx, img_idx = CENTER_PIC_ID)
    other_img   = visualize_equalized_hist(case_idx = opts.case_idx, img_idx = image_idx)
    
    kpts_cp, feats_cp, kpts_op, feats_op, matches \
        = get_coarse_matches(center_img, other_img, opts.case_idx, image_idx)
    F   = get_fundamental(opts.case_idx, CENTER_PIC_ID, image_idx)
    for step in range(opts.em_steps):
        can_output = (step == opts.em_steps - 1)
        weights, H, ransac_mask, original_mask = calculate_M(kpts_cp, feats_cp, kpts_op, feats_op, F, matches, opts, verbose = False, Hg = H_pred)           
        # To visualize the result of Affinity Matrix calculation, make verbose True
        
        if 'save' in opts.viz_kpt:
            viz_mask = ransac_mask                          # visualize RANSAC mask with spectral score
            if opts.viz == 'ransac':                        # visualize RANSAC mask
                viz_mask = original_mask
            elif opts.viz == 'weight_only':                 # visualize spectral score
                viz_mask = weights
            center_img_nc, other_img_nc = get_no_scat_img(opts.case_idx, image_idx, CENTER_PIC_ID)
            out_image = visualize_weighted(
                center_img_nc, other_img_nc, kpts_cp, kpts_op, matches, 
                viz_mask, mask_prev, draw_zero_w = True, only_diff = opts.only_diff
            )
            
            cv.imwrite(f"./{opts.base_folder}/case_{opts.case_idx}/kpts_viz_{image_idx}_em{step + 1}.png",  out_image)
            if opts.viz_kpt == 'save_quit' and can_output:  # do not exit until the last iteration         
                return                                      # do not solve the model, just visualize matches
        param = opts.huber_param if opts.lms else opts.fluc
        H_pred = model_solve(kpts_cp, kpts_op, matches, ransac_mask, param = param, max_iter = opts.max_iter, verbose = opts.verbose, lms = opts.lms)
        mask_prev = original_mask

        if opts.verbose:
            print("Ground truth homography: ", H.ravel())
        
        if opts.save_warpped and can_output:
            center_img_nc, other_img_nc = get_no_scat_img(opts.case_idx, image_idx, CENTER_PIC_ID)
            warpped_baseline = image_warping(center_img_nc, other_img_nc, H, False)
            warpped_result   = image_warping(center_img_nc, other_img_nc, H_pred, False)
            name = "lms" if opts.lms else "sdp"
            cv.imwrite(f"./{opts.base_folder}/case_{opts.case_idx}/{name}_{image_idx}.png",  warpped_result)
            cv.imwrite(f"./{opts.base_folder}/case_{opts.case_idx}/baseline_{image_idx}.png",warpped_baseline)

        if opts.save_hmat and can_output:
            H_save = np.linalg.inv(H_pred).astype(np.float64)       # Out put inversed
            H_save /= H_save[-1, -1]
            save2mat(f"case{opts.case_idx}/H3{image_idx}", H_save, name = 'H', prefix = "../diff_1/results/")
            if opts.baseline_hmat:
                H = np.linalg.inv(H).astype(np.float64)       # Out put inversed
                H /= H[-1, -1]
                save2mat(f"case{opts.case_idx}/H3{image_idx}_b", H, name = 'H', prefix = "../diff_1/results/")
    
""" 
    TODO: 
    - parameter fine tuning and parameter reading (from file)
    - Take another look at APAP
"""
if __name__ == "__main__":
    opts = get_options()
    spectral_method(opts)
    
    # Warning commented: /home/stn/.conda/envs/use/lib/python3.8/site-packages/cvxpy/problems/problem.py

