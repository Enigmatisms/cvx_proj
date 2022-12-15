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
import matplotlib.pyplot as plt

from itertools import compress
from utils import *

CENTER_PIC_ID = 3
PIC_LIST = (1, 2, 3, 4, 5)

def has_affinity(res_src_pts: list, pt: np.ndarray) -> int:
    for i, kpt in enumerate(res_src_pts):
        diff = np.abs((kpt - pt))
        if (diff < 1e-1).all():     # close enough
            return i
    return -1

# Read Rotation, Translation and Intrinsic Matrix


def match_all(case_idx: int, debug_disp: bool = False, verbose: bool = False):
    # Data preparation
    mat_file_path = f"{get_base_path(case_idx)}/keypoints.mat"
    feature_mat = scipy.io.loadmat(mat_file_path)["keypoints"]
    imgs = [visualize_equalized_hist(case_idx = case_idx, img_idx = idx) for idx in PIC_LIST]
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
        
        kpts_cp, _, kpts_op, _, matches = coarse_matching(center_img, other_img, raw_kpts_cp, raw_kpts_op)
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

def solve_3d_position(wf_origins: np.ndarray, wf_ray_dir: np.ndarray, verbose = False):
    As = np.eye(3) - wf_ray_dir @ np.transpose(wf_ray_dir, [0, 2, 1])
    p = np.sum(As @ wf_origins, axis = 0)
    A = np.sum(As, axis = 0)
    u, s, vt = np.linalg.svd(A)
    abs_last_v = abs(s[-1])
    if abs_last_v < 1e-3:
        if verbose:
            print(f"The last eigen value is too small: {abs_last_v}. Ill-conditioned matrix.") 
        s[-1] = s[-1] / abs_last_v * 1e-3 
    inv_A = vt.T @ np.diag(1 / s) @ u.T
    return inv_A @ p 
        
# Optimize 3D feature point using covisibility information 
def covid_3d_optimize(covid, res_src_pts, all_dst_pts, case_idx = 1, visualize = False):
    Rs, ts, K = read_rot_trans_int(f"{get_base_path(case_idx)}/Parameters.xlsx")
    K_inv = np.linalg.inv(K)
    for i, item in enumerate(covid): 
        if (np.int32(item) >= 0).sum() <= 1: continue
        
        pix_coords = [res_src_pts[i].reshape(2, 1)]     # put center coords first
        origins = [ts[CENTER_PIC_ID - 1].reshape(-1, 1)]
        M_rot = [Rs[CENTER_PIC_ID - 1]]
        cam_ids = [CENTER_PIC_ID]
        for j, idx in enumerate(item):
            if idx == -1: continue
            img_idx = j if j < CENTER_PIC_ID - 1 else j + 1
            
            pix_coords.append(all_dst_pts[j][idx].reshape(2, 1))
            # pix_coords.append(np.float32([[384], [384]]))
            M_rot.append(Rs[img_idx])
            # M_rot.append(Rs[img_idx])
            origins.append(ts[img_idx].reshape(-1, 1))
            # origins.append(ts[img_idx].reshape(-1, 1))
            cam_ids.append(img_idx + 1)
            # cam_ids.append(img_idx + 1)
        pix         = np.stack(pix_coords,  axis = 0)
        M_rots      = np.stack(M_rot,       axis = 0)
        wf_origins  = np.stack(origins,  axis = 0)
        wf_ray_dir  = world_frame_ray_dir(K_inv, pix, M_rots)
        dest_point  = wf_origins + 50 * wf_ray_dir
        solved_pt   = solve_3d_position(wf_origins, wf_ray_dir, True)
        print(f"Solved pt: {solved_pt.ravel()}")
        
        if visualize:
            plt.clf()
            plt.cla()
            ax = plt.figure().add_subplot(projection = '3d')
            for sp, ep, cam_id in zip(wf_origins, dest_point, cam_ids):
                ax.plot([sp[0, 0], ep[0, 0]], [sp[1, 0], ep[1, 0]], [sp[2, 0], ep[2, 0]], label = f'camera ray {cam_id}')
                ax.scatter(sp[0, 0], sp[1, 0], sp[2, 0])
            ax.scatter(solved_pt[0, 0], solved_pt[1, 0], solved_pt[2, 0], label = 'Estimated 3D point')
            ax.set_zlim((30, 85))
            ax.set_xlim((0, 50))
            ax.legend()
            plt.show()
    

if __name__ == "__main__":
    # path = f"{get_base_path(1)}/Parameters.xlsx"
    # read_rot_trans_int(path)
    case_idx = 1
    imgs, all_src_pts, all_dst_pts, all_matches = match_all(case_idx, False, False)
    res_src_pts, covid = get_covid(all_src_pts, all_matches)
    # visualize_covid(imgs, covid, res_src_pts, all_dst_pts, True)
    covid_3d_optimize(covid, res_src_pts, all_dst_pts, case_idx, True)