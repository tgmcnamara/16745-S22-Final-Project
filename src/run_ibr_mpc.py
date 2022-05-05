from cProfile import run
import numpy as np
import sys
from parse_network import parse_network
from build_QP_matrices import build_qp, update_qp 
from true_dynamics import DC_dynamics
from matplotlib import pyplot as plt

sys.path.append("..")
def run_ibr_mpc():
    # define simulation parameters
    tf = 10
    h = 0.05
    N = 10
    # play with these?
    omega_weight = 10.0
    rocof_weight = 10.0
    constrained = False

    # parse file and get initial state
    # 0 - 9 bus
    # 1 - 39 bus
    case = 0

    if case == 1:
        rawfile = 'ieee39_dc_solution.raw'
        jsonfile = 'ieee39_mpc_data.json'
    else:
        rawfile = 'case9_dc_sol.raw'
        jsonfile = 'case9_mpc_data.json'
    sim_data = parse_network(rawfile, jsonfile)
    n = len(sim_data['synch_gen'])
    m = len(sim_data['ibr'])    

    # assuming we're at steady state before disturbance,
    # so dtheta and domega are 0 initially
    x0 = np.zeros(2*n)

    # set up controller
    use_paper_ss = True
    qp, qp_info = build_qp(sim_data, x0, N, n, m, omega_weight, rocof_weight, h, constrained, use_paper_ss)

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
        x_next = DC_dynamics(x_prev, u, t, sim_data, qp_info)
        states_hist[:,t_ind+1] = x_next
        x_prev = np.copy(x_next)
        update_qp(qp, qp_info, sim_data, x_prev)

    # inspect and plot some results
    freq_base = 60
    plt.figure(1)
    if case == 1:
        freq_g4 = (states_hist[sim_data['synch_gen'][3].domega_index,:]+1)*freq_base
        freq_g9 = (states_hist[sim_data['synch_gen'][-1].domega_index,:]+1)*freq_base
        plt.plot(times, freq_g4, times, freq_g9)
    else:
        freq_g1 = (states_hist[sim_data['synch_gen'][0].domega_index,:]+1)*freq_base
        freq_g2 = (states_hist[sim_data['synch_gen'][1].domega_index,:]+1)*freq_base
        plt.plot(times, freq_g1, times, freq_g2)
    plt.xlabel('Simulation time (sec)')
    plt.ylabel('Generator frequency (Hz)')

    ibr0 = sim_data['ibr'][0]
    plt.figure(2)
    ibr_P = ibr0.P + np.array(ibr0.delP_hist)
    plt.plot(times[:-1], ibr_P)
    plt.xlabel('Simulation time (sec)')
    plt.ylabel('IBR output power (p.u.)')

    plt.show()


if __name__ == "__main__":
    run_ibr_mpc()