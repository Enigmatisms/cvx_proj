#-*-coding:utf-8-*-
"""
    LMS with optional Huber Loss
    @author: Qianyue He
    @date: 2022-12-16
"""

import time
import cvxpy as cp
import numpy as np
from numpy import ndarray as Arr

__all__ = ['LMSSolver']


class LMSSolver:
    def __init__(self, huber_param = -1.0) -> None:
        self.huber_param = huber_param
        self.h = cp.Variable((8, 1))    # homography

    @staticmethod
    def get_shifted(pts, num_points, x = 1.0):
        padded = np.concatenate([pts, np.repeat(np.float32([[x, 0, 0, 0]]), num_points, axis = 0)], axis = -1)
        rolled = np.roll(padded, shift = 3, axis = -1)
        front_part = np.concatenate([np.expand_dims(padded, axis = -2), np.expand_dims(rolled, axis = -2)], axis = -2)
        return front_part

    def solve(self, pts_c: Arr, pts_o: Arr, weights: Arr, verbose = 1, swap = True):
        num_points = pts_c.shape[0]
        rare_part = -pts_o[..., None] @ pts_c[:, None, :]        # shape (N, 2, 2)
        front_part = LMSSolver.get_shifted(pts_c, num_points)
        weights = np.repeat(weights[..., None], repeats = 2, axis = -1).reshape(-1, 1)      # make [1, 2, 3] -> [[1], [1], [2], [2], [3], [3]] -- shape (2N, 1)

        A = np.concatenate([front_part, rare_part], axis = -1).reshape(-1, 8) * weights # shape (2N, 8)
        rhs = pts_o.reshape(-1, 1) * weights
        if self.huber_param > 1e-2:         # valid huber param
            loss = 0
            diff = A @ self.h - rhs
            for item in diff:
                loss += cp.huber(item, self.huber_param)
            problem = cp.Problem(cp.Minimize(loss))
        else:
            problem = cp.Problem(cp.Minimize(cp.sum((A @ self.h - rhs) ** 2)))
        start_time = time.time()
        if verbose:
            print(f"Start solving... Huber Loss Used = [{self.huber_param > 1e-2}]")
        problem.solve()
        print(problem.value)
        end_time = time.time()
        solution = np.ones(9, dtype = np.float32)
        solution[:-1] = self.h.value.ravel()
        solution = solution.reshape(3, 3)
        
        if swap:
            solution = np.linalg.inv(solution)
        if verbose:
            print(f"Problem solved. Time consumption: {end_time - start_time:.3f}")
            print("The optimal value is", problem.value)
            print("Optimal solution:", self.h.value.ravel())
        return solution
    
def validation_test():
    pts_c = np.random.rand(20, 2)
    pts_o = np.random.rand(20, 2)
    weights = np.random.rand(20) * 0.7 + 0.3

    solver = LMSSolver()
    solver.solve(pts_c, pts_o, weights, verbose = True)
    

if __name__ == "__main__":
    validation_test()