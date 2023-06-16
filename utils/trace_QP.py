import numpy as np
import cvxpy as cp
import cvxopt


class EPO_LP_Trace(object):

    def __init__(self, m, n, privacy_constraint):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.n = n
        self.last_move = None
        self.a = cp.Parameter(m)        # Adjustments
        self.C = cp.Parameter((m, m))   # C: Gradient inner products, G^T G
        self.privacy_constraint = privacy_constraint

        self.beta = cp.Variable(m)     # Variable to optimize

        obj = cp.Minimize(cp.sum_squares(self.C @ self.beta - self.a))   # objective for balance
        constraints_rel = [self.beta >= -1, self.beta <= 1]  # privacy constraint not activated
        constraints_res = [self.beta >= -1, self.beta <= 1, self.C[1] @ self.beta >=0] # privacy constraint activated
        # constraints_res = [self.C[1] @ self.alpha >= 0]
        self.prob_rel = cp.Problem(obj, constraints_rel)  # LP balance
        self.prob_res = cp.Problem(obj, constraints_res)

        self.gamma = 0     # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0     # Stores the latest non-uniformity

    def get_alpha(self, l, G, C=False, mode=0):
        # mode=1: balance mode  mode=0: descent mode
        assert len(l) == len(G) == self.m, "length != m"
        self.mu_rl, self.a.value = anchor(l,mode)
        self.C.value = G if C else G @ G.T

        # the privacy constraint is not active
        if l[1] < self.privacy_constraint:
            self.gamma = self.prob_rel.solve(solver=cp.CVXOPT, verbose=False)
        else:
            self.gamma = self.prob_res.solve(solver=cp.CVXOPT, verbose=False)

        # print("last move: ", self.last_move)

        return self.beta.value, self.mu_rl


def anchor(l, mode):
    # r_reverse = 1/r
    r_reverse = np.array([0,1])
    l_norm = np.linalg.norm(l, 2)
    r_reverse_norm = np.linalg.norm(r_reverse, 2)
    l_dot_r = np.dot(l, r_reverse)
    # Cauchy-Schwarcz anchor
    mu = 1 - (l_dot_r**2 / (l_norm**2 * r_reverse_norm**2))
    if mode:
        a = (l_dot_r/(l_norm * r_reverse_norm)) * l - (l_norm/r_reverse_norm) * r_reverse
    else:
        a = l

    return mu, a
