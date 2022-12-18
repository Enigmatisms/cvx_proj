import argparse

def get_options(delayed_parse = False):
    # training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_idx", type = int, default = 1, help = "Id of Experiment (1, 2, 3, 4)")
    parser.add_argument("--img_idx",  type = int, default = 1, help = "Id of image (1, 2, 4, 5)")
    
    
    parser.add_argument("-v", "--verbose",      default = False, action = "store_true", help = "Output some intermediate information")
    parser.add_argument("-s", "--save_warpped", default = False, action = "store_true", help = "Save warpped images for visualization")
    parser.add_argument("--viz",                default = 'none', choices=['ransac', 'spectral', 'none'], help = "Visualization mode")

    # asymmetrical loss parameters
    parser.add_argument("--affinity_eps",   type = float, default = 30.0, help = "Sigma distance of allowed spatial inconsistency")
    parser.add_argument("--threshold",      type = float, default = 0.5, help = "Threshold for spectral score replacement")     # baseline 5.0
    parser.add_argument("--epi_weight",     type = float, default = 0.5, help = "Weighting coeff for epipolar score")
    parser.add_argument("--fluc",           type = float, default = 0.5, help = "SDP fluctuation parameter for robust solution")
    
    if delayed_parse:
        return parser
    return parser.parse_args()
