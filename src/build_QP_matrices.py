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
            rocof_weight,
            constrain_dPibr,
            use_paper_ss = False):
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
        self.constrain_dPibr = constrain_dPibr
        self.use_paper_ss = use_paper_ss

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
                P[(k*nnm+self.m):(k+1)*nnm, (k*nnm+self.m):(k+1)*nnm] = Q1
                # off diagonal terms for ROCOF
                P[k*nnm+self.m:(k+1)*nnm, ((k+1)*nnm+self.m):(k+2)*nnm] = Q2
                if k >= 1:
                    P[k*nnm+self.m:(k+1)*nnm, ((k-1)*nnm+self.m):k*nnm] = Q2
            else:
                P[(k*nnm+self.m):(k+1)*nnm, (k*nnm+self.m):(k+1)*nnm] = Q1f
                P[(k*nnm+self.m):(k+1)*nnm, (k-1)*nnm+self.m:k*nnm] = Q2
        return P
    
    def build_A(self, sim_data):
        gens = sim_data['synch_gen']
        m_vals = np.array([ele.inertia for ele in gens])
        d_vals = np.array([ele.damping for ele in gens])
        P_vals = np.array([ele.Pss for ele in gens])
        droop_vals = np.array([ele.droop for ele in gens])
        omega_nom = 120*np.pi # omega dynamics are in per unit
        
        # using the state space model presented in the ref. paper...
        Add = np.zeros((self.n, self.n))
        Adw = np.identity(self.n)
        Awd = np.diag(-1/m_vals) @ sim_data['Bgg']
        Aww = np.diag(1/m_vals*-d_vals)
        self.Apaper = np.block([[Aww, Awd], [Adw, Add]])

        # using the state space model that i think is correct...
        Add_true = np.identity(self.n)
        Adw_true = self.h*np.identity(self.n)*omega_nom
        Awd_true = np.diag(-self.h/m_vals) @ sim_data['Bgg']
        Aww_true = np.diag(1 + self.h/m_vals*(-d_vals))
        self.Atrue = np.block([[Aww_true, Awd_true], [Adw_true, Add_true]])
        if self.use_paper_ss:
            self.A = self.Apaper
        else:
            self.A = self.Atrue
        print("Eigenvalues of A:")
        print(np.linalg.eigvals(self.A))

    def build_B(self, sim_data):
        gens = sim_data['synch_gen']
        m_vals = np.array([ele.inertia for ele in gens])
        # effect of inputs (ibr angles)
        top_paper = np.diag(-1/m_vals) @ sim_data['Bgi']
        top_true = np.diag(-self.h/m_vals) @ sim_data['Bgi']
        self.Bu_paper = np.block([[top_paper], [np.zeros((self.n,self.m))]])
        self.Bu_true = np.block([[top_true], [np.zeros((self.n,self.m))]])
        # effect of disturbances
        self.Bd_paper = np.block([[np.diag(self.h/m_vals)], [np.zeros((self.n,self.n))]])
        self.Bd_true = np.block([[np.diag(self.h/m_vals)], [np.zeros((self.n,self.n))]])
        if self.use_paper_ss:
            self.Bu = self.Bu_paper
            self.Bd = self.Bd_paper
        else:
            self.Bu = self.Bu_true
            self.Bd = self.Bd_true

    def build_q(self, x0):
        q = np.zeros(self.N*(2*self.n+self.m))
        q[self.m:(self.m+self.n)] = -self.rocof_weight/self.h*x0[:self.n]
        return q

    def build_bounds(self, sim_data, x0):
        n = self.n
        nn = 2*n
        m = self.m
        if self.constrain_dPibr:
            ub = np.zeros(self.N*(nn+m))
            ub[:nn] = -self.A @ x0
            lb = np.copy(ub)
            dP_mins = np.zeros(m)
            dP_maxs = np.zeros(m)
            offset = self.N*nn
            for ele in sim_data['ibr']:
                dP_mins[ele.input_index] = ele.Pmin - ele.Pss
                dP_mins[ele.input_index] = ele.Pmax - ele.Pss
            for k in range(self.N-1):
                ub[(offset+k*m):(offset+(k+1)*m)] = dP_maxs
                lb[(offset+k*m):(offset+(k+1)*m)] = dP_mins
            # need to account part of dP_ibr0 term due to known gen angles
            ub[offset:offset+m] += -sim_data['Big'] @ x0[self.n:]
            lb[offset:offset+m] += -sim_data['Big'] @ x0[self.n:]
        else:
            ub = np.zeros(self.N*2*self.n)
            ub[:2*self.n] = -self.A @ x0
            lb = np.copy(ub)
        return (ub, lb)

    def build_C(self, sim_data):
        n = self.n
        nn = 2*n
        m = self.m
        mnn = nn+m
        # self.build_admittance(branches, gens, ibrs)
        self.build_A(sim_data)
        self.build_B(sim_data)
        Asp = sparse.csc_matrix(self.A)
        Bsp = sparse.csc_matrix(self.Bu)
        if self.constrain_dPibr:
            n_constraints = self.N*mnn
        else:    
            n_constraints = self.N*nn
        C = sparse.csc_matrix((n_constraints, self.N*(nn+m)))
        C[:nn, :m] = Bsp
        C[:nn, m:(m+nn)] = sparse.identity(nn)
        for k in range(1,self.N):
            C[k*nn:(k+1)*nn, (k-1)*mnn+m:k*mnn] = Asp
            C[k*nn:(k+1)*nn, k*mnn:k*mnn+m] = Bsp
            C[k*nn:(k+1)*nn, k*mnn+m:(k+1)*mnn] = -sparse.identity(nn)        
        if self.constrain_dPibr:
            offset = self.N*nn
            C[offset:(offset+m), :m] = sim_data['Bii']
            for k in range(1,self.N):
                C[(offset+k*m):(offset+(k+1)*m), (k-1)*mnn+m+n:k*mnn] = sim_data['Big']
                C[(offset+k*m):(offset+(k+1)*m), k*mnn:k*mnn+m] = sim_data['Bii']
        return C

def build_qp(sim_data, x0, N, n, m, omega_weight, rocof_weight, h, constrained, use_paper_ss):
    qp_info = QPinfo(n, m, N, h, omega_weight, rocof_weight, constrained, use_paper_ss)
    P = qp_info.build_P()
    q = qp_info.build_q(x0)
    C = qp_info.build_C(sim_data)
    (ub, lb) = qp_info.build_bounds(sim_data, x0)
    qp = osqp.OSQP()
    qp.setup(P=P, q=q, A=C, l=lb, u=ub)
    return qp, qp_info

def update_qp(qp, qp_info, sim_data, x0):
    q_new = qp_info.build_q(x0)
    (ub_new, lb_new) = qp_info.build_bounds(sim_data, x0)
    qp.update(q=q_new, l=lb_new, u=ub_new)