import numpy as np

def DC_dynamics(x, u, t, sim_data, qp_info):
    n = qp_info.n
    m = qp_info.m
    h = qp_info.h
    x_next = np.zeros(2*n)

    dtheta = np.zeros(n+m)
    dtheta[:n] = x[n:]
    dtheta[n:] = u

    ind_dyn = False
    if ind_dyn:
        dP = sim_data['Bhat'] @ dtheta
        for ele in sim_data['synch_gen']:
            dw = x[ele.domega_index]
            dth = x[ele.dtheta_index]
            Pl = dP[ele.domega_index]
            x_next[ele.domega_index] = dw + h*1/ele.inertia*(-ele.Pss*dw/ele.droop - Pl - ele.damping*dw - ele.check_dP(t))
            x_next[ele.dtheta_index] = dth + h*dw
        for ele in sim_data['ibr']:
            ele.delP_hist.append(dP[n+ele.input_index])
        
        # track dP for ibrs given the applied input
    else:
        delP_inv = sim_data['Big'] @ x[qp_info.n:] + sim_data['Bii'] @ u
        for ele in sim_data['ibr']:
            ele.delP_hist.append(delP_inv[ele.input_index])        
        # collect the disturbances
        delP = np.zeros(qp_info.n)
        for ele in sim_data['synch_gen']:
            delP[ele.domega_index] += ele.check_dP(t)
        
        x_next = (qp_info.A @ x) + (qp_info.Bu @ u) + (qp_info.Bd @ delP)
    return x_next

def powerflow_dynamics():
    pass

