import numpy as np

def DC_dynamics(x_prev, u, t, gens, ibrs, branches, qp_info):
    # matrix approach
    delP = np.zeros(len(gens))
    for ele in gens:
        delP[ele.domega_index] += ele.check_dP(t)
    x_next = (qp_info.A @ x_prev) + (qp_info.Bu @ u) + (qp_info.Bd @ delP)
    # loopy approach (allows incorporation of droops)

    return x_next

def powerflow_dynamics():
    pass

