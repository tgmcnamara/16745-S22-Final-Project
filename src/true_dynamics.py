import numpy as np

def DC_dynamics(x, u, t, sim_data, qp_info):
    n = qp_info.n
    ng = qp_info.ng
    m = qp_info.m
    h = qp_info.h
    x_next = np.zeros(2*n)

    # ind_dyn = False
    # dtheta = np.zeros(n+m)
    # dtheta[:n] = x[n:]
    # dtheta[n:] = u
    # if ind_dyn:
    #     dP = sim_data['Bhat'] @ dtheta
    #     for ele in sim_data['synch_gen']:
    #         dw = x[ele.domega_index]
    #         dth = x[ele.dtheta_index]
    #         Pl = dP[ele.domega_index]
    #         x_next[ele.domega_index] = dw + h*1/ele.inertia*(-ele.Pss*dw/ele.droop - Pl - ele.damping*dw - ele.check_dP(t))
    #         x_next[ele.dtheta_index] = dth + h*dw
    #     for ele in sim_data['ibr']:
    #         ele.delP_hist.append(dP[n+ele.input_index])
        
        # track dP for ibrs given the applied input
    # delP_inv = sim_data['Big'] @ x[:qp_info.ng] + sim_data['Bii'] @ u
    # for ele in sim_data['ibr']:
    #     ele.delP_hist.append(delP_inv[ele.input_index])        
    # collect the disturbances
    delP = np.zeros(ng)
    for ele in sim_data['synch_gen']:
        delP[ele.ddelta_index] += ele.check_dP(t)
    # turn off the disturbances for now
    delP = np.zeros(qp_info.ng)
    
    x_next = (qp_info.A @ x) + (qp_info.Bu @ u) + (qp_info.Bd @ delP)
    return x_next

def powerflow_dynamics():
    pass

