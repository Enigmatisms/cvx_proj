import argparse

def get_options(delayed_parse = False):
    # IO parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_idx", type = int, default = 1, help = "Id of Experiment (1, 2, 3, 4)")
    parser.add_argument("--img_idx",  type = int, default = 1, help = "Id of image (1, 2, 4, 5)")
    parser.add_argument("--em_steps", type = int, default = 2, help = "E-M step for iterative matching re-estimation")
    parser.add_argument("--max_iter", type = int, default = 8000, help = "Max number of iteration (should not be big)")

    parser.add_argument("--base_folder", type = str, default = "output", help = "Output base folder")
    
    parser.add_argument("-v", "--verbose",      default = False, action = "store_true", help = "Output some intermediate information")
    parser.add_argument("-s", "--save_warpped", default = False, action = "store_true", help = "Save warpped images for visualization")
    parser.add_argument("-m", "--save_hmat",    default = False, action = "store_true", help = "Save Homography mat for matlab evaluation")
    parser.add_argument("--lms",                default = False, action = "store_true", help = "Use LMS solver for the model")
    parser.add_argument("--only_diff",          default = False, action = "store_true", help = "Visualize only matching differences between EM steps")
    parser.add_argument("--baseline_hmat",      default = False, action = "store_true", help = "Whether to save OpenCV baseline Homography result")
    parser.add_argument("--viz_kpt",            default = 'none', choices=['save_quit', 'save', 'none'], help = "Visualize keypoint distribution")
    parser.add_argument("--viz",                default = 'ransac', choices=['ransac', 'spectral', 'weight_only'], help = "Visualization mode")

    # Prominent parameters that might affect training results
    parser.add_argument("--affinity_eps",   type = float, default = 30.0, help = "Sigma distance of allowed spatial inconsistency")
    parser.add_argument("--aff_thresh",     type = float, default = 0.5, help = "Threshold for spectral score replacement")     # baseline 5.0
    parser.add_argument("--epi_weight",     type = float, default = 0.5, help = "Weighting coeff for epipolar score")
    parser.add_argument("--fluc",           type = float, default = 0.5, help = "SDP fluctuation parameter for robust solution")
    parser.add_argument("--em_radius",      type = float, default = 6.0, help = "Point outside this radius after projection will not be considered")
    parser.add_argument("--score_thresh",   type = float, default = 0.4, help = "The matched features should have a similarity score above this threshold")
    
    parser.add_argument("--huber_param",          type = float, default = -1.0, help = "Huber Loss parameter. Value less than 0.01 means no Huber loss.")

    if delayed_parse:
        return parser
    return parser.parse_args()
