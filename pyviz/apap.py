"""
    APAP algorithm (modified). Credit to:
    EadCat: https://github.com/EadCat
    and Qianyue He
    @date 2022-12-17
"""

import cv2 as cv 
import numpy as np
import copy
from tqdm import tqdm


class APAP:
    def __init__(self, opt, final_size, offset):
        """
        :param opt: Engine running Options
        :param final_size: final result (Stitched) image size
        :param offset: The extent to which the image is stretched
        """
        super().__init__()
        self.gamma = opt.gamma
        self.sigma = opt.sigma
        self.final_width, self.final_height = final_size
        self.offset_x, self.offset_y = offset
        
    @staticmethod
    def getNormalize2DPts(point):
        """
        :param point: [sample_num, 2]
        :return:
        """
        sample_n, _ = point.shape
        origin_point = copy.deepcopy(point)
        # np.ones(6) [1, 1, 1, 1, 1, 1]
        padding = np.ones(sample_n, dtype=np.float)
        c = np.mean(point, axis=0)
        pt = point - c
        pt_square = np.square(pt)
        pt_sum = np.sum(pt_square, axis=1)
        pt_mean = np.mean(np.sqrt(pt_sum))
        scale = np.sqrt(2) / (pt_mean + 1e-8)
        # https://www.programmersought.com/article/9488862074/
        t = np.array([[scale, 0, -scale * c[0]],
                      [0, scale, -scale * c[1]],
                      [0, 0, 1]], dtype=np.float)
        origin_point = np.column_stack((origin_point, padding))
        new_point = t.dot(origin_point.T)
        new_point = new_point.T[:, :2]
        return t, new_point

    @staticmethod
    def getConditionerFromPts(point):
        sample_n, _ = point.shape
        calculate = np.expand_dims(point, 0)
        mean_pts, std_pts = cv.meanStdDev(calculate)
        mean_x, mean_y = np.squeeze(mean_pts)
        std_pts = np.squeeze(std_pts)
        std_pts = std_pts * std_pts * sample_n / (sample_n - 1)
        std_pts = np.sqrt(std_pts)
        std_x, std_y = std_pts
        std_x = std_x + (std_x == 0)
        std_y = std_y + (std_y == 0)
        norm_x = np.sqrt(2) / std_x
        norm_y = np.sqrt(2) / std_y
        T = np.array([[norm_x, 0, (-norm_x * mean_x)],
                      [0, norm_y, (-norm_y * mean_y)],
                      [0, 0, 1]], dtype=np.float)

        return T

    @staticmethod
    def point_normalize(nf, c):
        sample_n, _ = nf.shape
        cf = np.zeros_like(nf)

        for i in range(sample_n):
            cf[i, 0] = nf[i, 0] * c[0, 0] + c[0, 2]
            cf[i, 1] = nf[i, 1] * c[1, 1] + c[1, 2]

        return cf

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

        N1, nf1 = self.getNormalize2DPts(src_point)
        N2, nf2 = self.getNormalize2DPts(dst_point)

        C1 = self.getConditionerFromPts(nf1)
        C2 = self.getConditionerFromPts(nf2)

        cf1 = self.point_normalize(nf1, C1)
        cf2 = self.point_normalize(nf2, C2)

        inverse_sigma = 1. / (self.sigma ** 2)
        local_homography_ = np.zeros([mesh_n, pt_size, 3, 3], dtype=np.float)
        local_weight = np.zeros([mesh_n, pt_size, sample_n])
        aa = self.matrix_generate(sample_n, cf1, cf2)  # initiate A

        for i in range(mesh_n):
            for j in range(pt_size):

                dist = np.tile(vertices[i, j], (sample_n, 1)) - src_point
                weight = np.exp(-(np.sqrt(dist[:, 0] ** 2 + dist[:, 1] ** 2) * inverse_sigma))
                weight[weight < self.gamma] = self.gamma
                local_weight[i, j, :] = weight
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
        target = np.matmul(homography, pt)
        target /= target[2]
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

        for i in tqdm(range(self.final_height)) if progress else range(self.final_height):
            m = np.where(i < mesh_h)[0][0]
            for j in range(self.final_width):
                n = np.where(j < mesh_w)[0][0]
                homography = np.linalg.inv(local_homography[m-1, n-1, :])
                x, y = j - self.offset_x, i - self.offset_y
                source_pts = np.array([x, y, 1])
                target_pts = self.warp_coordinate_estimate(source_pts, homography)
                if 0 < target_pts[0] < ori_w and 0 < target_pts[1] < ori_h:
                    warped_img[i, j, :] = ori_img[int(target_pts[1]), int(target_pts[0]), :]

        return warped_img
