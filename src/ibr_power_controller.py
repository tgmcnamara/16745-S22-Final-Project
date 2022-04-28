import numpy as np
from parse_network import parse_network
from build_QP_matrices import build_qp, update_qp 
from true_dynamics import DC_dynamics

def run_ibr_mpc(casefile):
    # define simulation parameters
    tf = 10
    h = 0.05
    N = 10
    # play with these?
    omega_weight = 1.0
    rocof_weight = 1.0

    # parse file and get initial state
    (gens, ibrs, branches) = parse_network(casefile)
    n = len(gens)
    m = len(ibrs)    

    # get initial state (from DC powerflow maybe??)
    x0 = np.zeros(2*n)

    # set up controller
    qp, qp_info = build_qp(gens, ibrs, branches, x0, N, n, m, omega_weight, rocof_weight, h)

    # run simulation
    times = np.arange(0, tf+h, h)
    Nt = len(times)
    results_hist = {}
    states_hist = np.zeros((2*n, Nt))
    inputs_hist = np.zeros((m, Nt))
    states_hist[:,0] = x0
    x_prev = x0

    for t_ind in range(Nt-1):
        t = times[t_ind]
        results = qp.solve()
        results_hist[t] = results
        u = results.x[:m]
        inputs_hist[:,0] = u
        # ideally run with more accurate dynamics
        x_next = DC_dynamics(x_prev, u, gens, ibrs, branches, qp_info)
        states_hist[:,t_ind+1] = x_next
        x_prev = np.copy(x_next)
        update_qp(qp, qp_info, x_prev)

    # inspect and plot some results
