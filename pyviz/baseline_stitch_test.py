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

CENTER_PIC_ID = 3

__all__ = ['visualize_feature_pairs']
    
def visualize_feature_pairs(
    center_img: np.ndarray, other_img: np.ndarray, 
    case_idx: int = 1, pic_id: int = 1, 
    disp = False, swap = True, savemat = False
):
    out_image = np.concatenate((center_img, other_img), axis = 1)

    raw_kpts_cp, raw_kpts_op = get_features(case_idx, pic_id, CENTER_PIC_ID)
    kpts_cp, feats_cp, kpts_op, feats_op, matches = coarse_matching(center_img, other_img, raw_kpts_cp, raw_kpts_op)

    if savemat:
        save2mat('feats_cp', feats_cp)
        save2mat('feats_op', feats_op)
    
    print(f"Coarse matching result: {len(matches)}")
    
    out_img = cv.drawMatches(center_img, kpts_cp, other_img, kpts_op, matches, out_image, 1)
    if disp:
        imshow("key point matching", out_img)
        
    # params: contrastThreshold, edgeThreshold, sigma, descriptorType, see help(cv.SIFT)
    src_pts = np.float32([ kpts_cp[m.queryIdx].pt for m in matches ])
    dst_pts = np.float32([ kpts_op[m.trainIdx].pt for m in matches ])
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    
    if swap:
        H = np.linalg.inv(H)
    
    draw_params = dict(matchColor = (0, 255, 0),
                        singlePointColor = None,
                        matchesMask = mask.ravel().tolist(),
                        flags = 2
    )
    
    out_img = cv.drawMatches(center_img, kpts_cp, other_img, kpts_op, matches, out_image, **draw_params)
    print(f"Number of matches: {len(matches)}, valid matches: {mask.sum()}")
    if disp:
        imshow("key point RANSAC", out_img)
    cv.imwrite("./output/result.png", out_img)
    src_pts = src_pts[mask.ravel() > 0]
    dst_pts = dst_pts[mask.ravel() > 0]
    if swap:
        return dst_pts, src_pts, H
    return src_pts, dst_pts, H

def visualize_warpped_result(center_img: np.ndarray, other_img: np.ndarray, H: np.ndarray, direct_blend = False):
    warped = image_warping(other_img, center_img, H, direct_blend)
    imshow("warpped image", warped)
    cv.imwrite("./output/warpped_result.png", warped)
    
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
    _, _, H = visualize_feature_pairs(center_img, other_img, case_idx = case_idx, pic_id = img_idx, disp = True, savemat = True)
    
    center_img_nc, other_img_nc = get_no_scat_img(case_idx, img_idx, CENTER_PIC_ID)
    visualize_warpped_result(other_img_nc, center_img_nc, H, direct_blend = False)
    