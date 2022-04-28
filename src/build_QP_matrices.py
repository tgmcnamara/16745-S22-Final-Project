import numpy as np
from scipy import sparse
import osqp

class QPinfo:
    def __init__(
            self,
            n,
            m,
            N,
            h,
            omega_weight,
            rocof_weight):
        self.n = n
        self.m = m
        self.N = N
        self.h = h
        self.omega_weight = omega_weight
        self.rocof_weight = rocof_weight
        self.bgg = None
        self.bgi = None
        self.A = None
        self.B = None

    def build_P(self):
        nnm = 2*self.n + self.m
        Q1weights = np.concatenate(((self.omega_weight + 2/self.h*self.rocof_weight)*np.ones(self.n), np.zeros(self.n)))
        Q1 = sparse.diags(Q1weights)
        Q1f = sparse.diags(np.concatenate(((self.omega_weight + 1/self.h*self.rocof_weight)*np.ones(self.n), np.zeros(self.n))))
        Q2weights = np.concatenate(((-1/self.h*self.rocof_weight)*np.ones(self.n), np.zeros(self.n)))
        Q2 = sparse.diags(Q2weights)
        P = sparse.csc_matrix((self.N*nnm, self.N*nnm))
        for k in range(self.N):
            if k < (self.N-1):
                # main diagonal terms
                P[(k*nnm+self.m):(k+1)*nnm, k*nnm+self.m:(k+1)*nnm] = Q1
                # off diagonal terms for ROCOF
                P[k*nnm+self.m:(k+1)*nnm, (k+1)*nnm+self.m:(k+2)*nnm] = Q2
                if k >= 1:
                    P[k*nnm+self.m:k*nnm, (k-1)*nnm+self.m:k*nnm] = Q2
            else:
                P[(k*nnm+self.m):(k+1)*nnm, k*nnm+self.m:(k+1)*nnm] = Q1f
                P[k*nnm+self.m:k*nnm, (k-1)*nnm+self.m:k*nnm] = Q2
        return P

    def build_admittance(self, branches, gens, ibrs):
        bgg = np.zeros((self.n, self.n), dtype=float)
        bgi = np.zeros((self.n, self.m), dtype=float)
        for ele in branches:
            if ele.from_bus_gen and ele.to_bus_gen:
                bgg[ele.from_bus_index, ele.from_bus_index] += ele.b
                bgg[ele.from_bus_index, ele.to_bus_index] += -ele.b
                bgg[ele.to_bus_index, ele.from_bus_index] += -ele.b
                bgg[ele.to_bus_index, ele.to_bus_index] += -ele.b
            elif ele.from_bus_gen and ele.to_bus_ibr:
                bgg[ele.from_bus_index, ele.from_bus_index] += ele.b
                bgi[ele.from_bus_index, ele.to_bus_index] += -ele.b
            elif ele.to_bus_gen and ele.from_bus_ibr:
                bgg[ele.to_bus_index, ele.to_bus_index] += ele.b
                bgi[ele.to_bus_index, ele.from_bus_index] += -ele.b
        self.bgg = bgg
        self.bgi = bgi
        return bgg, bgi
    
    def build_A(self, gens):
        m_vals = np.array([ele.inertia for ele in gens])
        d_vals = np.array([ele.damping for ele in gens])
        top_left = np.diag(-d_vals/m_vals)
        top_right = (np.diag(-1/m_vals) @ self.bgg)
        self.A = np.block([[top_left, top_right], [np.identity(self.n, dtype=float), np.zeros((self.n,self.n), dtype=float)]])

    def build_B(self, gens, ibrs):
        n = len(gens)
        m_vals = np.array([ele.inertia for ele in gens])
        top = np.diag(-1/m_vals) @ self.bgi
        Bu = np.block([[top], [np.zeros(n,n)]])
        # disturbance term, ignored for now
        Bd = np.block([[np.diag(-1/m_vals)], [np.zeros(n,n)]])
        self.Bu = Bu
        self.Bd = Bd

    def build_q(self, x0):
        q = np.zeros(self.N*(2*self.n+self.m))
        q[self.m:(self.m+self.n)] = -self.rocof_weight/self.h*x0[:self.n] # dont use the angle states
        return q

    def build_bounds(self, x0):
        ub = np.zeros(self.N*2*self.n)
        ub[:2*self.n] = -self.A @ x0
        lb = np.copy(ub)
        return (ub, lb)

    def build_C(self, gens, ibrs, branches):
        nn = 2*self.n
        mnn = nn+self.m
        self.build_admittance(branches, gens, ibrs)
        self.build_A(gens)
        self.build_B(gens, ibrs)
        Asp = sparse.csc_matrix(self.A)
        Bsp = sparse.csc_matrix(self.Bu)
        C = sparse.csc_matrix((self.N*nn, self.N*(nn+self.m)))
        C[:nn, :self.m] = Bsp
        C[:nn, self.m:(self.m+nn)] = sparse.identity(nn)
        for k in range(1,self.N):
            C[k*nn:(k+1)*nn, (k-1)*mnn+self.m:k*mnn] = Asp
            C[k*nn:(k+1)*nn, k*mnn:k*mnn+self.m] = Bsp
            C[k*nn:(k+1)*nn, k*mnn+self.m:(k+1)*mnn] = -sparse.diags(nn)        
        return C

def build_qp(gens, ibrs, branches, x0, N, n, m, omega_weight, rocof_weight, h):
    qp_info = QPinfo(n, m, N, h, omega_weight, rocof_weight)
    P = qp_info.build_P()
    q = qp_info.build_q(x0)
    (C, A) = qp_info.build_C(gens, ibrs, branches)
    (ub, lb) = qp_info.build_bounds(x0)
    qp = osqp.OSQP()
    qp.setup(P=P, q=q, A=C, l=lb, u=ub)
    return qp, qp_info

def update_qp(qp, qp_info, x0):
    q_new = qp_info.build_q(x0)
    (ub_new, lb_new) = qp_info.build_bounds(x0)
    qp.update(q=q_new, l=lb_new, u=ub_new)