import numpy as np
import cvxpy as cp
import cvxopt


class Pareto_LP(object):

    def __init__(self, m, n):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.n = n
        self.C = cp.Parameter((m, m))   # C: Gradient inner products, G^T G
        self.rhs = cp.Parameter(m)      # RHS of constraints for balancing
        self.alpha = cp.Variable(m)     # Variable to optimize

        obj = cp.Maximize(cp.sum(self.alpha @ self.C))  # obj for descent

        constraints = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Relaxed
                           self.C @ self.alpha >= 0]
        self.prob = cp.Problem(obj, constraints)  # LP dominance

        self.gamma = 0     # Stores the latest Optimum value of the LP problem

    def get_alpha(self, l, G, C=False):
        print("solving alpha")
        #print("G: ",G)
        #print("l: ", l)
        #print("length: ",len(l),len(G),self.m)
        #assert len(l) == len(G) == len(r) == self.m, "length != m"
        self.C.value = G if C else G @ G.T
        #print("C: ",self.C.value)
        
        print("parameters ready")

        print("solving pareto optimality")
        self.gamma = self.prob.solve(solver=cp.GLPK, verbose=False)
        print("status:", self.prob.status)
        print("optimal value", self.prob.value)

        print("result: ",self.alpha.value)
        return self.alpha.value
