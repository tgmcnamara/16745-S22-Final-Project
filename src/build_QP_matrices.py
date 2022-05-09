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
        self.ni = len(sim_data['ibr'])
        self.N = N
        self.h = h
        self.omega_weight = omega_weight
        self.rocof_weight = rocof_weight
        self.dPibr_weight = dPibr_weight
        self.bgg = None
        self.bgi = None
        self.A = None
        self.Bu = None
        self.Bd = None
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
            Qul = 0.001*self.omega_weight*np.identity(ng)
            R = 0.001*self.omega_weight*np.identity(m)
        else:
            Qul = 0.001*self.omega_weight*np.identity(ng)
            R = self.dPibr_weight*np.identity(m)
        Qll = self.omega_weight*np.identity(ng)
        Q = np.block([[Qul, np.zeros((ng,ng))], [np.zeros((ng,ng)), Qll]])
        (self.K_dlqr, self.S_dlqr, self.E_dlqr) = control.dlqr(self.A, self.Bu, Q, R)
        return self.K_dlqr

    def build_P(self, sim_data):
        ng = self.ng
        ni = self.ni
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
        R = self.dPibr_weight * sparse.identity(m)
        # dPul = sparse.csc_matrix(self.dPibr_weight*(sim_data['Big'].transpose() @ sim_data['Big']))
        # dPur = sparse.csc_matrix(self.dPibr_weight*(sim_data['Big'].transpose() @ sim_data['Bii']))
        # dPll = sparse.csc_matrix(self.dPibr_weight*(sim_data['Bii'].transpose() @ sim_data['Big']))
        # dPlr = sparse.csc_matrix(self.dPibr_weight*(sim_data['Bii'].transpose() @ sim_data['Bii']))
        # diag_block = sparse.bmat([[dPul, None, dPur], [None, Qw_diag, None], [dPll, None, dPlr]])
        Qth_zeros = sparse.csc_matrix((ng,ng), dtype=float)
        diag_blocks = [Qth_zeros, Qw_diag]
        if self.include_Pm_droop:
            QPm_zeros = sparse.csc_matrix((ng,ng), dtype=float)
            diag_blocks.append(QPm_zeros)
        if self.constraint_mode == 2:
            QEbatt_zeros = sparse.csc_matrix((ni,ni), dtype=float)
            diag_blocks.append(QEbatt_zeros)
        diag_blocks.append(R)
        diag_block = sparse.block_diag(diag_blocks)
        P = sparse.csc_matrix((self.N*nm, self.N*nm))
        # terms that only apply to the 0th input
        P[:m,:m] = R
        for k in range(self.N-1):
                # main diagonal blocks
                P[(m+k*nm):(m+(k+1)*nm), (m+k*nm):(m+(k+1)*nm)] = diag_block
                # ROCOF off diagonal to the right
                P[(m+ng+k*nm):(m+2*ng+k*nm), (m+(k+1)*nm+ng):(m+(k+1)*nm+2*ng)] = Qw_offdiag
                # ROCOF off diagonal below
                P[(m+(k+1)*nm+ng):(m+(k+1)*nm+2*ng), (m+k*nm+ng):(m+k*nm+2*ng)] = Qw_offdiag
        # deal with the diagonal cost terms for the Nth state
        P[-ng:,-ng:] = Qw_diag_f
        return P

    def build_q(self, sim_data, x0):
        q = np.zeros(self.N*(self.n+self.m))
        # q[:self.m] = self.dPibr_weight*(sim_data['Big'] @ x0[:self.ng])
        q[(self.ng+self.m):(2*self.ng+self.m)] = -self.rocof_weight/(self.h**2)*x0[self.ng:]
        return q

    def build_A(self, sim_data):
        ng = self.ng
        h = self.h
        ni = self.ni
        gens = sim_data['synch_gen']
        m_vals = np.array([ele.inertia for ele in gens])
        d_vals = np.array([ele.damping for ele in gens])

        # using the state space model presented in the ref. paper...
        if self.use_paper_ss:
            Add = np.zeros((ng, ng))
            Adw = np.identity(ng)
            Awd = np.diag(-1/m_vals) @ sim_data['Bgg']
            Aww = np.diag(1/m_vals*-d_vals)
            self.A = np.block([[Add, Adw], [Awd, Aww]])
            return

        # omega states are in per unit (i.e. 1/(60Hz))
        # whereas dtheta states are in radians
        omega_nom = 120*np.pi
        Add = np.identity(ng)
        Adw = self.h*np.identity(ng)#*omega_nom
        Awd = np.diag(h/m_vals) @ (-sim_data['Bgg'] + sim_data['Bgi']@np.linalg.inv(sim_data['Bii'])@sim_data['Big'])
        Aww = np.diag(1 + h/m_vals*(-d_vals))
        A = np.block([[Add, Adw], [Awd, Aww]])
        # governor dynamics (if applicable)
        if self.include_Pm_droop:
            P_vals = np.array([ele.Pss for ele in gens])
            R_droop = np.array([ele.R_droop for ele in gens])
            K_droop = np.array([ele.K_droop for ele in gens])
            AdPm = np.zeros((ng,ng))
            AwPm = np.identity(h/m_vals)
            APmd = np.zeros((ng,ng))
            APmw = np.diagonal(-h*K_droop)
            APmPm = np.diagonal(1 - h*K_droop*R_droop)
            A_aug = np.zeros((3*ng,3*ng))
            A_aug[:2*ng,2*ng] = A
            A_aug[:ng,2*ng:] = AdPm
            A_aug[ng:2*ng,2*ng:] = AwPm
            A_aug[2*ng:,:ng] = APmd
            A_aug[2*ng:,ng:2*ng] = APmw
            A_aug[2*ng:,2*ng:] = APmPm
            A = A_aug
        # battery charge dynamics (if applicable)
        # simple assumption that dE/dt = -dP
        if self.constraint_mode == 2:
            AdE = np.zeros((ng,ni))
            AwE = np.zeros((ng,ni))
            AEd = np.zeros((ni,ng))
            AEw = np.zeros((ni,ng))
            AEE = np.identity(ni)
            if self.include_Pm_droop:
                APmE = np.zeros((ng,ni))
                AEPm = np.zeros((ni,ng))
                A_aug = np.zeros((3*ng+ni,3*ng+ni))
                A_aug[:3*ng,:3*ng] = A
                A_aug[:ng, 3*ng:] = AdE
                A_aug[ng:2*ng, 3*ng:] = AwE
                A_aug[2*ng:3*ng, 3*ng:] = APmE
                A_aug[3*ng:, :ng] = AEd
                A_aug[3*ng:, ng:2*ng] = AEw
                A_aug[3*ng:, 2*ng:3*ng] = AEPm
                A_aug[3*ng:, 3*ng:] = AEE
            else:
                A_aug = np.zeros((2*ng+ni,2*ng+ni))
                A_aug[:2*ng,:2*ng] = A
                A_aug[:ng, 2*ng:] = AdE
                A_aug[ng:2*ng, 2*ng:] = AwE
                A_aug[2*ng:, :ng] = AEd
                A_aug[2*ng:, ng:2*ng] = AEw
                A_aug[3*ng:, 3*ng:] = AEE
            A = A_aug
        self.A = A
        print("Eigenvalues of A:")
        print(np.linalg.eigvals(self.A))

    def build_B(self, sim_data):
        ng = self.ng
        ni = self.ni
        m = self.m
        gens = sim_data['synch_gen']
        m_vals = np.array([ele.inertia for ele in gens])
        if self.use_paper_ss:
            # effect of inputs (ibr angles)
            self.Bu = np.block([[np.zeros((ng,m))], [np.diag(-1/m_vals) @ sim_data['Bgi']]])
            # effect of disturbances
            self.Bd = np.block([[np.zeros((ng, ng))], [np.diag(1/m_vals)]])
            return
        Bu = np.block([[np.zeros((ng,m))], [np.diag(-self.h/m_vals) @ sim_data['Bgi']@np.linalg.inv(sim_data['Bii'])]])
        Bd = np.block([[np.zeros((ng, ng))], [np.diag(self.h/m_vals)]])
        if self.include_Pm_droop:
            Bu = np.block([[Bu], [np.zeros((ng,m))]])
        if self.constraint_mode == 2:
            Bu = np.block([[Bu], [-self.h*np.identity(ni)]])
        self.Bu = Bu
        self.Bd = Bd

    def build_bounds(self, sim_data, x0):
        n = self.n
        ng = self.ng
        ni = self.ni
        m = self.m
        ub = np.zeros(self.N*n)
        ub[:self.n] = -self.A @ x0
        lb = np.copy(ub)
        offset = self.N*n
        if self.include_Pm_droop:
            ub = np.append(ub, np.zeros(self.N*ng))
            lb = np.append(lb, np.zeros(self.N*ng))
            dPg_mins = np.zeros(ng)
            dPg_maxs = np.zeros(ng)
            for ele in sim_data['synch_gen']:
                dPg_mins[ele.state_index] = ele.Pmin - ele.Pss
                dPg_maxs[ele.state_index] = ele.Pmax - ele.Pss
            for k in range(self.N-1):
                lb[(offset+k*ng):(offset+(k+1)*ng)] = dPg_mins
                ub[(offset+k*ng):(offset+(k+1)*ng)] = dPg_maxs
            offset += self.N*ng
        if self.constraint_mode == 1:
            # just dPmin ≤ i ≤ dPmax
            ub = np.append(ub, np.zeros(self.N*m))
            lb = np.append(lb, np.zeros(self.N*m))
            dPi_mins = np.zeros(m)
            dPi_maxs = np.zeros(m)
            for ele in sim_data['ibr']:
                dPi_mins[ele.input_index] = ele.Pmin - ele.Pss
                dPi_maxs[ele.input_index] = ele.Pmax - ele.Pss
            for k in range(self.N-1):
                lb[(offset+k*m):(offset+(k+1)*m)] = dPi_mins
                ub[(offset+k*m):(offset+(k+1)*m)] = dPi_maxs
        elif self.constraint_mode == 2:
            # TODO - battery charge constraints
            ub = np.append(ub, np.zeros(self.N*ni))
            lb = np.append(lb, np.zeros(self.N*ni))
            Ebatt_mins = np.zeros(ni)
            Ebatt_maxs = np.zeros(ni)
            for ele in sim_data['ibr']:
                Ebatt_mins[ele.input_index] = ele.Ebatt_min
                Ebatt_maxs[ele.input_index] = ele.Ebatt_max
            for k in range(self.N):
                lb[(offset+k*ni):(offset+(k+1)*ni)] = dPi_mins
                ub[(offset+k*ni):(offset+(k+1)*ni)] = dPi_maxs
            
        return (ub, lb)

    def build_C(self, sim_data):
        n = self.n
        ng = self.ng
        ni = self.ni
        m = self.m
        mn = n+m
        # self.build_admittance(branches, gens, ibrs)
        self.build_A(sim_data)
        self.build_B(sim_data)
        Asp = sparse.csc_matrix(self.A)
        Bsp = sparse.csc_matrix(self.Bu)
        n_constraints = self.N*n
        if self.include_Pm_droop:
            n_constraints += ng
        if self.constraint_mode == 1:
            n_constraints += m
        elif self.constraint_mode == 2:
            n_constraints += ni
        C = sparse.csc_matrix((n_constraints, self.N*(mn)))
        C[:n, :m] = Bsp
        C[:n, m:(mn)] = -sparse.identity(n)
        for k in range(1,self.N):
            C[k*n:(k+1)*n, m+(k-1)*mn:k*mn] = Asp
            C[k*n:(k+1)*n, k*mn:k*mn+m] = Bsp
            C[k*n:(k+1)*n, m+k*mn:(k+1)*mn] = -sparse.identity(n)        
        offset = self.N*n
        if self.include_Pm_droop:
            # just 1's in the dPm indices for each time step
            for k in range(self.N):
                C[(offset+k*ng):(offset+(k+1)*ng), (m+2*ng+k*mn):(offset+(k+1)*ng)] = sparse.identity(ng)
            offset += self.N*ng
        if self.constraint_mode == 1:
            # just 1's in the u indices for each time step
            # C[offset:(offset+m), :m] = sparse.identity(m)
            for k in range(self.N):
                C[(offset+k*m):(offset+(k+1)*m), k*mn:k*mn+m] = sparse.identity(m)
        elif self.constraint_mode == 2:
            # just 1s on the Ebatt indices for each time step
            for k in range(self.N):
                C[(offset+k*ni):(offset+(k+1)*ni), k*mn:k*mn+m] = sparse.identity(m)
        return C

def build_qp(sim_data, x0, N, h, omega_weight, rocof_weight, dPibr_weight, constraint_mode, include_Pm_droop, use_paper_ss):
    qp_info = QPinfo(sim_data, x0, N, h, omega_weight, rocof_weight, dPibr_weight, constraint_mode, include_Pm_droop)
    P = qp_info.build_P(sim_data)
    q = qp_info.build_q(sim_data, x0)
    C = qp_info.build_C(sim_data)
    (ub, lb) = qp_info.build_bounds(sim_data, x0)
    qp = osqp.OSQP()
    qp.setup(P=P, q=q, A=C, l=lb, u=ub)
    return qp, qp_info

def update_qp(qp, qp_info, sim_data, x0):
    q_new = qp_info.build_q(sim_data, x0)
    (ub_new, lb_new) = qp_info.build_bounds(sim_data, x0)
    qp.update(q=q_new, l=lb_new, u=ub_new)

def build_lqr(sim_data, x0, N, h, omega_weight, rocof_weight, dPibr_weight, constraint_mode, include_Pm_droop, use_paper_ss=False):
    qp_info = QPinfo(sim_data, x0, N, h, omega_weight, rocof_weight, dPibr_weight, constraint_mode, include_Pm_droop, use_paper_ss)
    K = qp_info.build_LQR(sim_data)
    return K, qp_info