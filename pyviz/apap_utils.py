#-*-coding:utf-8-*-
"""
    The code in here is from repo: https://github.com/EadCat/APAP-Image-Stitching
    Slightly revised by Qianyue He
    @date 2022-12.18
"""

import numpy as np

def get_mesh(size, mesh_size, start=0):
    """
    :param size: final size [width, height]
    :param mesh_size: # of mesh
    :param start: default 0
    :return:
    """
    w, h = size
    x = np.linspace(start, w, mesh_size)
    y = np.linspace(start, h, mesh_size)

    return np.stack([x, y], axis=0)

def get_vertice(size, mesh_size, offsets):
    """
    :param size: final size [width, height]
    :param mesh_size: # of mesh
    :param offsets: [offset_x, offset_y]
    :return:
    """
    w, h = size
    x = np.linspace(0, w, mesh_size)
    y = np.linspace(0, h, mesh_size)
    next_x = x + w / (mesh_size * 2)
    next_y = y + h / (mesh_size * 2)
    next_x, next_y = np.meshgrid(next_x, next_y)
    vertices = np.stack([next_x, next_y], axis=-1)
    vertices -= np.array(offsets)
    return vertices

def final_size(src_img, dst_img, project_H):
    """
    get the size of stretched (stitched) image
    :param src_img: source image
    :param dst_img: destination image
    :param project_H: global homography
    :return:
    """
    h, w, _ = src_img.shape

    corners = []
    pt_list = [np.float32([0, 0, 1]), np.float32([0, h, 1]),
               np.float32([w, 0, 1]), np.float32([w, h, 1])]

    for pt in pt_list:
        vec = np.matmul(project_H, pt)
        x, y = vec[0] / vec[2], vec[1] / vec[2]
        corners.append([x, y])

    corners = np.array(corners).astype(np.int)

    h, w, _ = dst_img.shape

    max_x = max(np.max(corners[:, 0]), w)
    max_y = max(np.max(corners[:, 1]), h)
    min_x = min(np.min(corners[:, 0]), 0)
    min_y = min(np.min(corners[:, 1]), 0)

    width = max_x - min_x
    height = max_y - min_y
    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0

    return width, height, offset_x, offset_y

def uniform_blend(img1, img2):
    # grayscale
    gray1 = np.mean(img1, axis=-1)
    gray2 = np.mean(img2, axis=-1)
    result = (img1.astype(np.float64) + img2.astype(np.float64))

    g1, g2 = gray1 > 0, gray2 > 0
    g = g1 & g2
    mask = np.expand_dims(g * 0.5, axis=-1)
    mask = np.tile(mask, [1, 1, 3])
    mask[mask == 0] = 1
    result *= mask
    result = result.astype(np.uint8)
    return result
