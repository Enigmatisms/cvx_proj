#-*-coding:utf-8-*-
"""
    Utility functions
    @author: Qianyue He
    @date: 2022-12-13
"""

import cv2 as cv
import numpy as np

def get_base_path(case_idx: int = 1):
    return f"../diff_1/raw_data/case{case_idx}"

def get_path(case_idx: int = 1, img_idx: int = 1, no_scat = False):
    return f"{get_base_path(case_idx)}/{'no_' if no_scat else ''}scat/img_{'no' if no_scat else ''}haze{img_idx}.png"

def imshow(name: str, pic: np.ndarray):
    print("Press any key to quit.")
    while True:
        cv.imshow("equalized hist", pic)
        key = cv.waitKey(30)
        if key > 0:
            cv.destroyAllWindows()
            return key == 27
    
def visualize_equalized_hist(case_idx = 1, img_idx = 3, disp = False):
    img_path = get_path(case_idx, img_idx)
    img = cv.imread(img_path)
    img_eq = np.stack([cv.equalizeHist(img[..., i]) for i in range(3)], axis = -1)
    if disp:
        imshow("equalized hist", img_eq)
    return img_eq