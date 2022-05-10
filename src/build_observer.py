import numpy as np
from build_QP_matrices import QPinfo
import control

class observer:
    def __init__(
        self,
        qp_info,
        C,
        Cd
    ):
        self.A = qp_info.A
        self.Bu = qp_info.Bu
        self.Bd = qp_info.Bd
        self.C = C
        self.Cd = Cd
        self.n = qp_info.n
        self.m = qp_info.m
        self.ng = qp_info.ng
        self.ni = qp_info.ni
        self.nd = qp_info.ng

    def check_system_observable(self):
        # Need to confirm that the matrix:
        # | A-I  Bd |
        # |  C   Cd |
        # has full column rank
        ul = self.A - np.identity(self.n)
        ur = self.Bd
        ll = self.C
        lr = self.Cd
        check_matrix = np.block([[ul, ur], [ll, lr]])
        return (np.linalg.matrix_rank(check_matrix) == check_matrix.shape[1])

    def place_poles(self, state_poles, disturbance_poles):
        self.Lx = control.place(self.A.transpose(), self.C.transpose(), state_poles).transpose()
        self.Ld = control.place(self.A.transpose(), self.C.transpose(), state_poles).transpose()
        self.obsv_dyn_A = np.block([[(self.A + self.Lx@self.C), (self.Bd + self.Lx@self.Cd)],
                               [(self.Ld@self.C), (np.identity(self.nd) + self.Ld@self.Cd)]])
        self.obsv_dyn_B = np.block([[self.Bu], [np.zeros((self.nd, self.m))]])
        self.obsv_dyn_L = np.block([[self.Lx], [self.Ld]])

    def observer_dynamics(self, xhat_prev, dhat_prev, u, y, t):
        z = np.concatenate((xhat_prev, dhat_prev))
        z_next = (self.obsv_dyn_A @ z) + (self.obsv_dyn_B @ u) - (self.obsv_dyn_L @ y)
        xhat_next = z_next[:self.n]
        dhat_next = z_next[self.n:]
        return xhat_next, dhat_next
