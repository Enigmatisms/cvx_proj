#-*-coding:utf-8-*-
"""
    APAP algorithm (modified). Credit to:
    EadCat: https://github.com/EadCat
    and Qianyue He
    @date 2022-12-17
"""

import copy
import cv2 as cv 
import numpy as np

from sys import argv
from tqdm import tqdm
from apap_utils import *
from utils import save2mat
from baseline_stitch_test import visualize_feature_pairs
from utils import imshow, visualize_equalized_hist, get_no_scat_img


class APAP:
    def __init__(self, gamma, sigma, final_size, offset):
        """
        :param opt: Engine running Options
        :param final_size: final result (Stitched) image size
        :param offset: The extent to which the image is stretched
        """
        super().__init__()
        self.gamma = gamma
        self.sigma = sigma
        self.final_width, self.final_height = final_size
        self.offset_x, self.offset_y = offset
        
    @staticmethod
    def getNormalize2DPts(point):
        """
        Function translates and normalises a set of 2D homogeneous points so that
        their centroid is at the origin and their mean distance from the origin is sqrt(2)
        :param point: [sample_num, 2]
        :return:
        """
        sample_n, _ = point.shape
        origin_point = copy.deepcopy(point)
        # np.ones(6) [1, 1, 1, 1, 1, 1]
        padding = np.ones(sample_n, dtype=np.float32)
        c = np.mean(point, axis=0)
        pt = point - c                                  # decentralize
        pt_square = np.square(pt)                       # variance in each axis
        pt_sum = np.sum(pt_square, axis=1)              # variance for on point
        pt_mean = np.mean(np.sqrt(pt_sum))              # std for one point -> mean std for all points
        scale = np.sqrt(2) / (pt_mean + 1e-8)           # sqrt(2) / std
        # https://www.programmersought.com/article/9488862074/
        t = np.array([[scale, 0, -scale * c[0]],        # translation and normalization
                      [0, scale, -scale * c[1]],
                      [0, 0, 1]], dtype=np.float32)
        origin_point = np.column_stack((origin_point, padding))
        new_point = t.dot(origin_point.T)
        new_point = new_point.T[:, :2]                  # remove homogeneous part 1.
        return t, new_point

    # input is of shape (N, 2)
    @staticmethod
    def getConditionerFromPts(point):
        """
        Function translates and normalises a set of 2D homogeneous points so that
        their centroid is at the origin and their mean distance from the origin is sqrt(2).
        
        Basically, the functionality is the similar to getNormalize2DPts
        """
        sample_n, _ = point.shape
        # calculate = np.expand_dims(point, 0)            # (1, N , 2)
        # mean_pts, std_pts = cv.meanStdDev(calculate)
        mean_pts = np.mean(point, axis = 0)
        std_pts  = np.std(point, axis = 0)
        mean_x, mean_y = mean_pts
        # std_pts = np.squeeze(std_pts)                     # no need to squeeze
        
        std_pts = std_pts * std_pts * sample_n / (sample_n - 1)     # variance is n/(n-1) * estimated variance of samples
        std_pts = np.sqrt(std_pts)                          # std
        std_x, std_y = std_pts
        std_x = std_x + (std_x == 0)
        std_y = std_y + (std_y == 0)
        norm_x = np.sqrt(2) / std_x
        norm_y = np.sqrt(2) / std_y
        T = np.array([[norm_x, 0, (-norm_x * mean_x)],
                      [0, norm_y, (-norm_y * mean_y)],
                      [0, 0, 1]], dtype=np.float32)

        return T

    @staticmethod
    def point_normalize(nf, c):
        sample_n, _ = nf.shape
        cf = np.zeros_like(nf)

        for i in range(sample_n):       # inefficient matrix mul
            cf[i, 0] = nf[i, 0] * c[0, 0] + c[0, 2]
            cf[i, 1] = nf[i, 1] * c[1, 1] + c[1, 2]

        return cf
    
    @staticmethod
    def matrix_generate(sample_n, cf1, cf2):
        A = np.zeros([sample_n * 2, 9], dtype=np.float32)
        for k in range(sample_n):
            A[2 * k, 0] = cf1[k, 0]
            A[2 * k, 1] = cf1[k, 1]
            A[2 * k, 2] = 1
            A[2 * k, 6] = (-cf2[k, 0]) * cf1[k, 0]
            A[2 * k, 7] = (-cf2[k, 0]) * cf1[k, 1]
            A[2 * k, 8] = (-cf2[k, 0])

            A[2 * k + 1, 3] = cf1[k, 0]
            A[2 * k + 1, 4] = cf1[k, 1]
            A[2 * k + 1, 5] = 1
            A[2 * k + 1, 6] = (-cf2[k, 1]) * cf1[k, 0]
            A[2 * k + 1, 7] = (-cf2[k, 1]) * cf1[k, 1]
            A[2 * k + 1, 8] = (-cf2[k, 1])
        return A

    def local_homography(self, src_point, dst_point, vertices):
        """
        local homography estimation
        :param src_point: shape [sample_n, 2]
        :param dst_point:
        :param vertices: shape [mesh_size, mesh_size, 2]
        :return: np.ndarray [meshsize, meshsize, 3, 3]
        """
        sample_n, _ = src_point.shape
        mesh_n, pt_size, _ = vertices.shape

        # N1, N2, normalization and translation matrix
        N1, nf1 = self.getNormalize2DPts(src_point)
        N2, nf2 = self.getNormalize2DPts(dst_point)

        C1 = self.getConditionerFromPts(nf1)
        C2 = self.getConditionerFromPts(nf2)

        cf1 = self.point_normalize(nf1, C1)
        cf2 = self.point_normalize(nf2, C2)

        inverse_sigma = 1. / (self.sigma ** 2)
        local_homography_ = np.zeros([mesh_n, pt_size, 3, 3], dtype=np.float32)
        local_weight = np.zeros([mesh_n, pt_size, sample_n])
        aa = self.matrix_generate(sample_n, cf1, cf2)  # initiate A
        
        for i in range(mesh_n):
            for j in range(pt_size):

                dist = np.tile(vertices[i, j], (sample_n, 1)) - src_point
                weight = np.exp(-(np.sqrt(dist[:, 0] ** 2 + dist[:, 1] ** 2) * inverse_sigma))
                weight[weight < self.gamma] = self.gamma
                local_weight[i, j, :] = weight
                
                # what we should do is to replace this place
                # if we are to compute one SDP at a time, which will be extremely time consuming
                # multi-processing acceleration might be used
                # re-weighting the rows using distance
                A = np.expand_dims(np.repeat(weight, 2), -1) * aa
                _, _, V = cv.SVDecomp(A)
                h = V[-1, :]
                
                
                h = h.reshape((3, 3))
                h = np.linalg.inv(C2).dot(h).dot(C1)
                h = np.linalg.inv(N2).dot(h).dot(N1)
                h = h / h[2, 2]
                local_homography_[i, j] = h
        return local_homography_, local_weight

    @staticmethod
    def warp_coordinate_estimate(pt, homography):
        """
        source points -> target points matrix multiplication with homography
        [ h11 h12 h13 ] [x]   [x']
        [ h21 h22 h23 ] [y] = [y']
        [ h31 h32 h33 ] [1]   [s']
        :param pt: source point
        :param homography: transfer relationship
        :return: target point
        """
        target = homography @ pt
        target /= target[2]         # homogeneous coordinates
        return target

    def local_warp(self, ori_img: np.ndarray, local_homography: np.ndarray, mesh: np.ndarray,
                   progress=False) -> np.ndarray:
        """
        this method requires improvement with the numpy algorithm (because of speed)
        :param ori_img: original input image
        :param local_homography: [mesh_n, pt_size, 3, 3] local homographies np.ndarray
        :param mesh: [2, mesh_n+1]
        :param progress: print warping progress or not.
        :return: result(warped) image
        """
        mesh_w, mesh_h = mesh
        ori_h, ori_w, _ = ori_img.shape
        warped_img = np.zeros([self.final_height, self.final_width, 3], dtype=np.uint8)
        mesh_n, pt_size, _, _ = local_homography.shape
        print("Inverse solving started.")
        for i in range(mesh_n):
            for j in range(pt_size):
                local_homography[i, j, :] = np.linalg.inv(local_homography[i, j, :])
        print("Inverse solving completed.")

        for i in tqdm(range(self.final_height)) if progress else range(self.final_height):
            m = np.where(i < mesh_h)[0][0]
            for j in range(self.final_width):
                n = np.where(j < mesh_w)[0][0]
                homography = local_homography[m-1, n-1, :]
                x, y = j - self.offset_x, i - self.offset_y
                source_pts = np.array([x, y, 1])
                target_pts = self.warp_coordinate_estimate(source_pts, homography)
                if 0 < target_pts[0] < ori_w and 0 < target_pts[1] < ori_h:
                    warped_img[i, j, :] = ori_img[int(target_pts[1]), int(target_pts[0]), :]

        return warped_img

    
if __name__ == "__main__":
    mesh_size   = 100
    gamma       = 0.5
    sigma       = 100
    
    argv_len = len(argv)
    case_idx = 1
    img_idx = 1
    CENTER_PIC_ID = 3
    if argv_len > 1:
        case_idx = int(argv[1]) 
    if argv_len > 2:
        img_idx = int(argv[2]) 
        
    center_img = visualize_equalized_hist(case_idx = case_idx, img_idx = CENTER_PIC_ID)
    other_img = visualize_equalized_hist(case_idx = case_idx, img_idx = img_idx)
    final_src, final_dst, H = visualize_feature_pairs(center_img, other_img, case_idx = case_idx, pic_id = img_idx, swap = True)
    
    final_w, final_h, offset_x, offset_y = final_size(center_img, other_img, H)
    mesh = get_mesh((final_w, final_h), mesh_size + 1)
    vertices = get_vertice((final_w, final_h), mesh_size, (offset_x, offset_y))
    stitcher = APAP(gamma, sigma, [final_w, final_h], [offset_x, offset_y])
    local_homography, local_weight = stitcher.local_homography(final_src, final_dst, vertices)

    # After estimation: blending
    center_img_nc, other_img_nc = get_no_scat_img(case_idx, img_idx, CENTER_PIC_ID)
    cimg_h, c_imgw, _ = center_img_nc.shape

    print(f"local_homography shape: {local_homography.shape}")

    mesh_y, mesh_x, _, _ = local_homography.shape
    for i in range(mesh_y):
        for j in range(mesh_x):
            local_homography[i, j] = np.linalg.inv(local_homography[i, j].copy())
            local_homography[i, j] /= local_homography[i, j, -1, -1]


    
    # warpped_img = stitcher.local_warp(other_img_nc, local_homography, mesh, True)
    # dst_temp = np.zeros_like(warpped_img)
    # dst_temp[offset_y: cimg_h + offset_y, offset_x: c_imgw + offset_x, :] = center_img_nc
    # result = uniform_blend(warpped_img, dst_temp)
    # cv.imwrite("./output/apap.png", result)
    local_homography = local_homography.transpose(0, 1, 3, 2)
    local_homography = local_homography.astype(np.float64).reshape(-1, 9)
    save2mat(f"case{case_idx}/H3{img_idx}_apap", local_homography, name = 'H', prefix = "../diff_1/results/")