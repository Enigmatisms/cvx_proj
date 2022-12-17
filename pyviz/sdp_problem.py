import time
import cvxpy as cp
import numpy as np
from numpy import ndarray as Arr

class SDPSolver:
    def __init__(self, du = 1.0, dv = 1.0) -> None:
        self.du = du
        self.dv = dv

        self.r = cp.Variable(1)         # lambda (lagrange)
        self.t = cp.Variable(1)         # epigraph
        self.h = cp.Variable((8, 1))    # homography


    @staticmethod
    def get_shifted(pts, num_points, x = 1.0):
        padded = np.concatenate([pts, np.repeat(np.float32([[x, 0, 0, 0]]), num_points, axis = 0)], axis = -1)
        rolled = np.roll(padded, shift = 3, axis = -1)
        front_part = np.concatenate([np.expand_dims(padded, axis = -2), np.expand_dims(rolled, axis = -2)], axis = -2)
        return front_part
    
    """
        Create a semi-positive definite matrix constraint
    """
    def solve(self, pts_c: Arr, pts_o: Arr, weights: Arr, verbose = 1):
        # First: re-arrange pts_c to a reasonable structure
        num_points = pts_c.shape[0]
        rare_part = -pts_o[..., None] @ pts_c[:, None, :]        # shape (N, 2, 2)
        front_part = SDPSolver.get_shifted(pts_c, num_points)
        weights = np.repeat(weights[..., None], repeats = 2, axis = -1).reshape(-1, 1)      # make [1, 2, 3] -> [[1], [1], [2], [2], [3], [3]] -- shape (2N, 1)

        A = np.concatenate([front_part, rare_part], axis = -1).reshape(-1, 8) * weights # shape (2N, 8)
        rhs = pts_o.reshape(-1, 1) * weights

        u_pmatrix = np.repeat(np.float32([[self.du, 0]]), repeats = num_points, axis = 0)
        u_rare    = np.hstack((-pts_o.reshape(-1, 1), np.zeros((num_points << 1, 1))))
        u_front   = SDPSolver.get_shifted(u_pmatrix, num_points, 0.0).reshape(-1, 6)
        A1 = np.concatenate([u_front, u_rare], axis = -1) * weights
        
        v_pmatrix = np.repeat(np.float32([[0, self.dv]]), repeats = num_points, axis = 0)
        v_rare    = np.hstack((np.zeros((num_points << 1, 1)), -pts_o.reshape(-1, 1)))
        v_front   = SDPSolver.get_shifted(v_pmatrix, num_points, 0.0).reshape(-1, 6)
        A2 = np.concatenate([v_front, v_rare], axis = -1) * weights

        Px = cp.hstack((A1 @ self.h, A2 @ self.h))
        qx = (A @ self.h - rhs)
        Pxqx = cp.hstack((Px, qx))

        last_zero = np.eye(3)
        last_zero[-1, -1] = 0
        last_one = np.zeros((3, 3))
        last_one[-1, -1] = 1
        M = self.r * last_zero + self.t * last_one 

        upper_part = cp.hstack((np.eye(num_points << 1), Pxqx))
        lower_part = cp.hstack((Pxqx.T, M))
        constraint_matrix = cp.vstack((upper_part, lower_part))

        if verbose > 1:
            print(constraint_matrix)
        problem = cp.Problem(cp.Minimize(self.t + self.r), [constraint_matrix >> 0])
        start_time = time.time()
        if verbose:
            print("Start solving...")
        problem.solve()
        if verbose:
            print(f"Problem solved. Time consumption: {time.time() - start_time:.3f}")
            print("The optimal value is", problem.value)
            print("Optimal solution:", self.h.value.ravel())
        solution = np.ones(9, dtype = np.float32)
        solution[:-1] = self.h.value.ravel()
        return solution.reshape(3, 3)
    
def validation_test():
    pts_c = np.random.rand(20, 2)
    pts_o = np.random.rand(20, 2)
    weights = np.random.rand(20) * 0.7 + 0.3

    solver = SDPSolver()
    solver.solve(pts_c, pts_o, weights, verbose = True)
    

if __name__ == "__main__":
    validation_test()