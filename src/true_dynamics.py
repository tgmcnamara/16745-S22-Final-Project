import numpy as np

def DC_dynamics(x, u, d, t, qp_info):
    x_next = (qp_info.A @ x) + (qp_info.Bu @ u) + (qp_info.Bd @ d)
    return x_next

