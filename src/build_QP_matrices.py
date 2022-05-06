import numpy as np
from scipy import sparse
import osqp
import control

class QPinfo:
    def __init__(self,
                 sim_data,
                 x0,
                 N,
                 h,
                 omega_weight,
                 rocof_weight,
                 dPibr_weight,
                 constraint_mode,
                 include_Pm_droop,
                 use_paper_ss = False):
        self.n = sim_data['n']
        self.m = sim_data['m']
        self.ng = len(sim_data['synch_gen'])
        self.ni = len(sim_data['synch_gen'])
        self.N = N
        self.h = h
        self.omega_weight = omega_weight
        self.rocof_weight = rocof_weight
        self.dPibr_weight = dPibr_weight
        self.bgg = None
        self.bgi = None
        self.A = None
        self.B = None
        # constraint mode:
        # 0 - no constraints
        # 1 - upper and lower bounds on dP for IBRs
        # 2 - battery energy relation to dP for IBRs
        self.constraint_mode = constraint_mode

        self.include_Pm_droop = include_Pm_droop

        # allow using the incorrect state space model from the referenced paper
        self.use_paper_ss = use_paper_ss

    def build_LQR(self, sim_data):
        # create a closed-loop policy u = -K*x
        # for comparison to MPC
        # doesn't allow constraints or ROCOF in cost function
        m = self.m
        ng = self.ng
        ni = self.ni
        # generate A and B if they haven't been made yet
        if not self.A:
            self.build_A(sim_data)
        if not self.Bu:
            self.build_B(sim_data)
        # generate a positive definite Q and R
        if self.dPibr_weight == 0:
            # some nominal weight on inputs so R is pos. def.
            Qul = 0.01*self.omega_weight*np.identity(ng)
            R = 0.01*self.omega_weight*np.identity(m)
            N = None
        else:
            Qul = self.dPibr_weight*(sim_data['Big'].transpose @ sim_data['Big'])
            R = self.dPibr_weight*(sim_data['Bii'].transpose @ sim_data['Bii'])
            N = self.dPibr_weight*np.block([[sim_data['Big']], np.zeros((ni,ng))])
        Qll = self.omega_weight*np.identity(ng)
        Q = np.block([[Qul, np.zeros((ng,ng))], [np.zeros((ng,ng)), Qll]])
        (self.K_dlqr, self.S_dlqr, self.E_dlqr) = control.dlqr(self.A, self.B, Q, R, N)
        return self.K_dlqr

    def build_P(self, sim_data):
        ng = self.ng
        m = self.m
        nm = self.n + m
        h2 = self.h**2
        # diagonal freq costs (include domega and ROCOF costs)
        Qw_diag_weights = (self.omega_weight + 2/h2*self.rocof_weight)*np.ones(ng)
        # n x n in the omega,omega block
        Qw_diag = sparse.diags(Qw_diag_weights)
        Qw_diag_f_weights = (self.omega_weight + 1/h2*self.rocof_weight)*np.ones(ng)
        Qw_diag_f = sparse.diags(Qw_diag_f_weights)
        # off diagonal costs for ROCOF
        Qwoff_weights = (-1/h2*self.rocof_weight)*np.ones(ng)
        Qw_offdiag = sparse.diags(Qwoff_weights)
        dPul = sparse.csc_matrix(self.dPibr_weight*(sim_data['Big'].transpose @ sim_data['Big']))
        dPur = sparse.csc_matrix(self.dPibr_weight*(sim_data['Big'].transpose @ sim_data['Bii']))
        dPll = sparse.csc_matrix(self.dPibr_weight*(sim_data['Bii'].transpose @ sim_data['Big']))
        dPlr = sparse.csc_matrix(self.dPibr_weight*(sim_data['Bii'].transpose @ sim_data['Bii']))
        diag_block = sparse.bmat([[dPul, None, dPur], [None, Qw_diag, None], [dPll, None, dPlr]])
        P = sparse.csc_matrix((self.N*nm, self.N*nm))
        # terms that only apply to the 0th input
        P[:m,:m] = dPlr
        for k in range(self.N-1):
                # main diagonal blocks
                P[(m+k*nm):(m+(k+1)*nm), (m+k*nm):(m+(k+1)*nm)] = diag_block
                # ROCOF off diagonal to the right
                P[(m+ng+k*nm):((k+1)*nm), (m+ng+(k+1)*nm):((k+2)*nm)] = Qw_offdiag
                # ROCOF off diagonal below
                P[(m+ng+(k+1)*nm):((k+2)*nm), (m+ng+k*nm):((k+1)*nm)] = Qw_offdiag
        # deal with the diagonal cost terms for the Nth state
        P[-ng:,-ng:] = Qw_diag_f
        return P

    def build_q(self, sim_data, x0):
        q = np.zeros(self.N*(2*self.n+self.m))
        q[:self.m] = self.dPibr_weight*(self.sim_data['Big'] @ x0[:self.ng])
        q[self.m:(self.ng+self.m)] = -1/(self.h**2)*x0[self.ng:]
        return q

    def build_A(self, sim_data):
        ng = self.ng
        h = self.h
        gens = sim_data['synch_gen']
        m_vals = np.array([ele.inertia for ele in gens])
        d_vals = np.array([ele.damping for ele in gens])

        # using the state space model presented in the ref. paper...
        if self.use_paper_ss:
            Add = np.zeros((ng, ng))
            Adw = np.identity(ng)
            Awd = np.diag(-1/m_vals) @ sim_data['Bgg']
            Aww = np.diag(1/m_vals*-d_vals)
            self.A = np.block([[Adw, Add], [Aww, Awd]])
            return

        omega_nom = 120*np.pi # omega dynamics are in per unit
        Add = np.identity(ng)
        Adw = self.h*np.identity(ng)*omega_nom
        Awd = np.diag(-h/m_vals) @ sim_data['Bgg']
        Aww = np.diag(1 + h/m_vals*(-d_vals))
        if self.include_Pm_droop:
            P_vals = np.array([ele.Pss for ele in gens])
            R_droop = np.array([ele.R_droop for ele in gens])
            K_droop = np.array([ele.K_droop for ele in gens])
            AdPm = np.zeros((ng,ng))
            AwPm = np.identity(h/m_vals)
            APmd = np.zeros((ng,ng))
            APmw = np.diagonal(-h*K_droop)
            APmPm = np.diagonal(1 - h*K_droop*R_droop)
            self.A = np.block([[Add, Adw, AdPm], [Awd, Aww, AwPm], [APmd, APmw, APmPm]])
        else:
            self.A = np.block([[Add, Adw], [Awd, Aww]])
        print("Eigenvalues of A:")
        print(np.linalg.eigvals(self.A))

    def build_B(self, sim_data):
        ng = self.ng
        m = self.m
        gens = sim_data['synch_gen']
        m_vals = np.array([ele.inertia for ele in gens])
        if self.use_paper_ss:
            # effect of inputs (ibr angles)
            self.Bu = np.block([[np.zeros((ng,m))], [np.diag(-1/m_vals) @ sim_data['Bgi']]])
            # effect of disturbances
            self.Bd = np.block([[np.zeros((ng,ng))],[np.diag(self.h/m_vals)]]) 
        else:
            self.Bu = np.block([[np.zeros((ng,m))], [np.diag(-self.h/m_vals) @ sim_data['Bgi']]])
            self.Bd = np.block([[np.zeros((ng, ng))], [[np.diag(self.h/m_vals)]]])

    def build_bounds(self, sim_data, x0):
        n = self.n
        ng = self.ng
        m = self.m
        if self.constraint_mode == 1:
            ub = np.zeros(self.N*(n+m))
            ub[:n] = -self.A @ x0
            lb = np.copy(ub)
            dP_mins = np.zeros(m)
            dP_maxs = np.zeros(m)
            offset = self.N*n
            for ele in sim_data['ibr']:
                dP_mins[ele.input_index] = ele.Pmin - ele.Pss
                dP_mins[ele.input_index] = ele.Pmax - ele.Pss
            for k in range(self.N-1):
                ub[(offset+k*m):(offset+(k+1)*m)] = dP_maxs
                lb[(offset+k*m):(offset+(k+1)*m)] = dP_mins
            # need to account part of dP_ibr0 term due to known gen angles
            ub[offset:offset+m] += -sim_data['Big'] @ x0[self.n:]
            lb[offset:offset+m] += -sim_data['Big'] @ x0[self.n:]
        elif self.constraint_mode == 2:
            pass
            # TODO - battery constraints
        else:
            ub = np.zeros(self.N*n)
            ub[:self.n] = -self.A @ x0
            lb = np.copy(ub)
        return (ub, lb)

    def build_C(self, sim_data):
        n = self.n
        ng = self.ng
        m = self.m
        mn = n+m
        # self.build_admittance(branches, gens, ibrs)
        self.build_A(sim_data)
        self.build_B(sim_data)
        Asp = sparse.csc_matrix(self.A)
        Bsp = sparse.csc_matrix(self.Bu)
        if self.constraint_mode == 1:
            n_constraints = self.N*mn
        else:
            n_constraints = self.N*n
        C = sparse.csc_matrix((n_constraints, self.N*(mn)))
        C[:n, :m] = Bsp
        C[:n, m:(mn)] = -sparse.identity(n)
        for k in range(1,self.N):
            C[k*n:(k+1)*n, (k-1)*mn+m:k*mn] = Asp
            C[k*n:(k+1)*n, k*mn:k*mn+m] = Bsp
            C[k*n:(k+1)*n, k*mn+m:(k+1)*mn] = -sparse.identity(n)        
        if self.constraint_mode == 1:
            offset = self.N*n
            C[offset:(offset+m), :m] = sim_data['Bii']
            for k in range(1,self.N):
                C[(offset+k*m):(offset+(k+1)*m), (k-1)*mn+m+ng:k*mn] = sim_data['Big']
                C[(offset+k*m):(offset+(k+1)*m), k*mn:k*mn+m] = sim_data['Bii']
        elif self.constraint_mode == 2:
            # TODO - battery constraints
            pass
        return C

def build_qp(sim_data, x0, N, h, omega_weight, rocof_weight, dPibr_weight, constraint_mode, include_Pm_droop):
    qp_info = QPinfo(sim_data, x0, N, h, omega_weight, rocof_weight, dPibr_weight, constraint_mode, include_Pm_droop)
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

def build_lqr(sim_data, x0, N, h, omega_weight, rocof_weight, dPibr_weight, constraint_mode, include_Pm_droop):
    qp_info = QPinfo(sim_data, x0, N, h, omega_weight, rocof_weight, dPibr_weight, constraint_mode, include_Pm_droop)
    K = qp_info.build_LQR(sim_data)
    return K