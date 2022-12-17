import numpy as np
import cvxpy as cp

def modeling_test():
    Px = cp.Variable((3, 5))
    qx = cp.Variable((3, 1))
    last_zero = np.eye(6)
    last_zero[-1, -1] = 0
    last_one = np.zeros((6, 6))
    last_one[-1, -1] = 1
    lamb = cp.Variable(1)
    t = cp.Variable(1)
    M = lamb * last_zero + t * last_one 

    Pq = cp.hstack((Px, qx))
    upper = cp.hstack((np.eye(3), Px, qx))
    lower = cp.hstack((Pq.T, M))
    A = cp.vstack((upper, lower))
    constraint = A >> 0
    print(constraint)

if __name__ == '__main__':
    modeling_test()