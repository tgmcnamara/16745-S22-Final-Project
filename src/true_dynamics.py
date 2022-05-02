import numpy as np

def DC_dynamics(x_prev, u, t, sim_data, qp_info):
    # matrix approach
    delP = np.zeros(qp_info.n)
    for ele in sim_data['gens']:
        delP[ele.domega_index] += ele.check_dP(t)
    x_next = (qp_info.A @ x_prev) + (qp_info.Bu @ u) + (qp_info.Bd @ delP)
    return x_next

def powerflow_dynamics():
    pass

