#-*-coding:utf-8-*-
"""
    Python visualization of SIFT-RANSAC baseline blendering (Ghost Effect) 
    This is the baseline model
    @author: Qianyue He
    @date: 2022-12-12
"""

import scipy.io
import cv2 as cv
import numpy as np
from sys import argv
from utils import *

CENTER_PIC_ID = 3
    
def get_no_scat_img(case_idx, img_idx, center_id = CENTER_PIC_ID):
    img_path = get_path(case_idx, img_idx, True)
    ctr_path = get_path(case_idx, center_id, True)
    return cv.imread(ctr_path), cv.imread(img_path)
    
def visualize_feature_pairs(center_img: np.ndarray, other_img: np.ndarray, case_idx: int = 1, pic_id: int = 1, disp = False):
    out_image = np.concatenate((center_img, other_img), axis = 1)

    raw_kpts_cp, raw_kpts_op = get_features(case_idx, pic_id, CENTER_PIC_ID)
    kpts_cp, _, kpts_op, _, matches = coarse_matching(center_img, other_img, raw_kpts_cp, raw_kpts_op)
    
    print(f"Coarse matching result: {len(matches)}")
    
    out_img = cv.drawMatches(center_img, kpts_cp, other_img, kpts_op, matches, out_image, 1)
    if disp:
        imshow("key point matching", out_img)
        
    # params: contrastThreshold, edgeThreshold, sigma, descriptorType, see help(cv.SIFT)
    src_pts = np.float32([ kpts_cp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpts_op[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    
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
    return H

def image_warping(img_base: np.ndarray, img2warp: np.ndarray, H: np.ndarray, direct_blend = True):
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
                    result[row_id, col_id] = ((img_base[y_base, x_base].astype(int) + result[row_id, col_id].astype(int)) / 2).astype(np.uint8)
                else:                               # empty: superposition
                    result[row_id, col_id] = img_base[y_base, x_base]
    else:
        result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img_base
    return result

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
    H = visualize_feature_pairs(center_img, other_img, case_idx = case_idx, pic_id = img_idx, disp = True)
    
    center_img_nc, other_img_nc = get_no_scat_img(case_idx, img_idx, CENTER_PIC_ID)
    visualize_warpped_result(center_img_nc, other_img_nc, H, direct_blend = False)
    