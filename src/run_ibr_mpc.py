from cProfile import run
import numpy as np
import sys
from parse_network import parse_network
from build_QP_matrices import build_qp, update_qp 
from true_dynamics import DC_dynamics
from matplotlib import pyplot as plt

# TODO
# - generate closed loop policy using DLQR
# - start from small perturbation and see if it can just get it to origin
# - Kalman observer for disturbance estimation

sys.path.append("..")
def run_ibr_controller():
    # define simulation parameters
    tf = 30
    h = 0.05
    init_at_ss = False
    # penalize frequency deviation from 60 Hz
    omega_weight = 10.0
    # penalize rate of change of frequency deviation
    rocof_weight = 10.0
    # penalize use of IBRs (can leave at 0)
    dPibr_weight = 0.0

    # model the synch gen's governor droop controller in state space model
    include_Pm_droop = False

    # select whether to use a simple LQR controller 
    use_lqr = False
    # time step horizon for MPC, if applicable
    N = 10

    # constraint mode:
    # 0 - no constraints
    # 1 - upper and lower bounds on dP for IBRs
    # 2 - battery energy relation to dP for IBRs
    constraint_mode = 0
    
    # parse file and get initial state
    # 0 - 9 bus, 3 gen (1 IBR)
    # 1 - 39 bus, 10 gen (2 IBR)
    case = 0

    if case == 1:
        rawfile = 'ieee39_dc_solution.raw'
        jsonfile = 'ieee39_mpc_data.json'
    else:
        rawfile = 'case9_dc_sol.raw'
        jsonfile = 'case9_mpc_data.json'
    sim_data = parse_network(rawfile, jsonfile, constraint_mode, include_Pm_droop)
    n = sim_data['n']
    m = sim_data['m']
    ngen = len(sim_data['synch_gen'])
    nibr = len(sim_data['ibr'])

    # assuming we're at steady state before disturbance,
    # so dtheta and domega are 0 initially
    x0 = np.zeros(n)
    if not init_at_ss:
        # initial angle deviations
        # 0?
        # initial freq deviations
        x0[ngen:2*ngen] = -.01 + 0.001*np.random.randn(ngen)

    # set up controller
    qp, qp_info = build_qp(sim_data, x0, N, h, omega_weight, rocof_weight, dPibr_weight, constraint_mode, include_Pm_droop)

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