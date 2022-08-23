import numpy as np
import cvxpy as cp
import cvxopt


class EPO_LP(object):

    def __init__(self, m, n, r, eps=1e-3):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.n = n
        self.r = r
        self.eps = eps
        self.last_move = None
        self.a = cp.Parameter(m)        # Adjustments
        self.C = cp.Parameter((m, m))   # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(m)       # d_bal^TG
        self.rhs = cp.Parameter(m)      # RHS of constraints for balancing

        self.alpha = cp.Variable(m)     # Variable to optimize

        obj_bal = cp.Maximize(self.alpha @ self.Ca)   # objective for balance
        constraints_bal = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Simplex
                           self.C @ self.alpha >= self.rhs]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance

        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))  # obj for descent
        constraints_res = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Restrict
                           self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
                           self.C @ self.alpha >= 0]
        constraints_rel = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Relaxed
                           self.C @ self.alpha >= 0]
        self.prob_dom = cp.Problem(obj_dom, constraints_res)  # LP dominance
        self.prob_rel = cp.Problem(obj_dom, constraints_rel)  # LP dominance

        self.gamma = 0     # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0     # Stores the latest non-uniformity

    def get_alpha(self, l, G, r=None, C=False, relax=True,allow_privacy_ascent = False):
        print("solving alpha")
        r = self.r if r is None else r
        #print("r: ",r)
        #print("G: ",G)
        #print("length: ",len(l),len(G),len(r),self.m)
        assert len(l) == len(G) == len(r) == self.m, "length != m"
        rl, self.mu_rl, self.a.value = adjustments(l, r)
        #print("rl: ",rl)
        self.C.value = G if C else G @ G.T
        #print("C: ",self.C.value)
        self.Ca.value = self.C.value @ self.a.value
        #print("Ca: ",self.Ca.value)
        
        #print("parameters ready")
        

        if self.mu_rl > self.eps:
            #print("reducing non-uniformity")
            J = self.Ca.value > 0
            #print("J set: ",J)
            if len(np.where(J)[0]) > 0:
                #print("J set is not empty")
                J_star_idx = np.where(rl == np.max(rl))[0]
                #print("J star index: ",J_star_idx)
                self.rhs.value = self.Ca.value.copy()
                self.rhs.value[J] = -np.inf     # Not efficient; but works.
                self.rhs.value[J_star_idx] = 0
                if allow_privacy_ascent == False:
                    self.rhs.value[1] = 0  # privacy loss is not allowed to ascent
            else:
                #print("J set is empty")
                self.rhs.value = np.zeros_like(self.Ca.value)
            self.gamma = self.prob_bal.solve(solver=cp.GLPK, verbose=False)
            #print("status:", self.prob_bal.status)
            #print("optimal value", self.prob_bal.value)
            # self.gamma = self.prob_bal.solve(verbose=False)
            self.last_move = "bal"
            #print("last move: balance")
        else:
            #print("solving pareto optimality")
            if relax:
                #print("relax problem")
                self.gamma = self.prob_rel.solve(solver=cp.GLPK, verbose=False)
                #print("status:", self.prob_rel.status)
                #print("optimal value", self.prob_rel.value)
            else:
                #print("restrict problem")
                self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
                #print("status:", self.prob_dom.status)
                #print("optimal value", self.prob_dom.value)

            # self.gamma = self.prob_dom.solve(verbose=False)
            self.last_move = "dom"
            #print("last move: dominance")

        #print("result: ",self.alpha.value)
        return self.alpha.value


def mu(rl, normed=False):
    if len(np.where(rl < 0)[0]):
        raise ValueError(f"rl<0 \n rl={rl}")
        return None
    m = len(rl)
    l_hat = rl if normed else rl / rl.sum()
    eps = np.finfo(rl.dtype).eps
    l_hat = l_hat[l_hat > eps]
    return np.sum(l_hat * np.log(l_hat * m))


def adjustments(l, r):
    #print("adjustments")
    m = len(l)
    #print("m: ",m)
    #print("r: ",r)

    rl = r * l
    #print("rl: ",rl)
    l_hat = rl / rl.sum()
    #print("l hat: ",l_hat)
    mu_rl = mu(l_hat, normed=True)
    #print("mu_rl: ",mu_rl)
    a = r * (np.log(l_hat * m) - mu_rl)
    #print("a: ",a)
    return rl, mu_rl, a
