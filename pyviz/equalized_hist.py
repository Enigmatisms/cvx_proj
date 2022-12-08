import cv2 as cv
import numpy as np
from sys import argv
import scipy.io

def get_base_path(case_idx: int = 1):
    return f"../diff_1/raw_data/case{case_idx}"

def get_path(case_idx: int = 1, img_idx: int = 1):
    return f"{get_base_path(case_idx)}/scat/img_haze{img_idx}.png"

CENTER_PIC_ID = 3

def visualize_equalized_hist(case_idx = 1, img_idx = 3, disp = False):
    img_path = get_path(case_idx, img_idx)
    img = cv.imread(img_path)
    img_eq = np.stack([cv.equalizeHist(img[..., i]) for i in range(3)], axis = -1)
    if disp:
        print("Press any key to quit.")
        while True:
            cv.imshow("equalized hist", img_eq)
            key = cv.waitKey(30)
            if key > 0:
                break
        cv.destroyAllWindows()
    return img_eq
    
def visualize_feature_pairs(center_img: np.ndarray, other_img: np.ndarray, case_idx: int = 1, pic_id: int = 1, disp = False):
    mat_file_path = f"{get_base_path(case_idx)}/keypoints.mat"
    feature_mat = scipy.io.loadmat(mat_file_path)["keypoints"]
    # print(feature_mat)
    valid_pic_id = {1, 2, 3, 4, 5}
    valid_pic_id.remove(CENTER_PIC_ID)
    if not pic_id in valid_pic_id:
        raise ValueError(f"{pic_id} not valid (3 is the id of the center image, therefore [1, 2, 4, 5] are available)")

    feat_mat = feature_mat[pic_id - 1][0]
    raw_kpts_op = feat_mat[3:-1, :].T              # raw keypoints of non-center pictures, discarding the homogeneous dim
    raw_kpts_cp = feat_mat[:2, :].T                # raw keypoints of the center pictures, discarding the homogeneous dim

    out_image = np.concatenate((center_img, other_img), axis = 1)
    kpts_cp = [cv.KeyPoint(*pt, 1) for pt in raw_kpts_cp]
    kpts_op = [cv.KeyPoint(*pt, 1) for pt in raw_kpts_op]
    matches = []
    out_img = cv.drawMatches(center_img, kpts_cp, other_img, kpts_op, matches, out_image, 1)
    if disp:
        print("Press any key to quit.")
        while True:
            cv.imshow("key point matching", out_img)
            key = cv.waitKey(30)
            if key > 0:
                break
        cv.destroyAllWindows()
    # params: contrastThreshold, edgeThreshold, sigma, descriptorType, see help(cv.SIFT)
    extractor = cv.SIFT.create(32)
    (kpts_cp, feats_cp) = extractor.compute(center_img, kpts_cp)
    (kpts_op, feats_op) = extractor.compute(other_img, kpts_op)
    macther = cv.FlannBasedMatcher()
    matches = macther.match(feats_cp, feats_op)
    
    src_pts = np.float32([ kpts_cp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpts_op[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    
    draw_params = dict(matchColor = (0, 255, 0),
                        singlePointColor = None,
                        matchesMask = mask.ravel().tolist(),
                        flags = 2
    )
    
    out_img = cv.drawMatches(center_img, kpts_cp, other_img, kpts_op, matches, out_image, **draw_params)
    if disp:
        print("Press any key to quit.")
        while True:
            cv.imshow("key point matching", out_img)
            key = cv.waitKey(30)
            if key > 0:
                break
        cv.destroyAllWindows()
    cv.imwrite("result.png", out_img)
    
if __name__ == "__main__":
    # visualize_equalized_hist()
    argv_len = len(argv)
    case_idx = 1
    img_idx = 1
    if argv_len > 1:
        case_idx = int(argv[1]) 
    if argv_len > 2:
        img_idx = int(argv[2]) 
        
    center_img = visualize_equalized_hist()
    other_img = visualize_equalized_hist(img_idx = img_idx)
    visualize_feature_pairs(center_img, other_img, case_idx = case_idx, pic_id = img_idx, disp = True)