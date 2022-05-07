from cProfile import run
import numpy as np
import sys
from parse_network import parse_network
from build_QP_matrices import build_qp, update_qp, build_lqr
from true_dynamics import DC_dynamics
from matplotlib import pyplot as plt

# TODO
# - generate closed loop policy using DLQR
# - start from small perturbation and see if it can just get it to origin
# - Kalman observer for disturbance estimation

sys.path.append("..")
def run_ibr_controller():
    # define simulation parameters
    tf = 10
    h = 0.05
    init_at_ss = False
    # penalize frequency deviation from 60 Hz
    omega_weight = 1.0
    # penalize rate of change of frequency deviation
    rocof_weight = 1.0
    # penalize use of IBRs (can leave at 0)
    dPibr_weight = 0.0

    # use steady state model from original paper for comparison
    # (it is very much the wrong model)
    use_paper_ss = False
    
    # model the synch gen's governor droop controller in state space model
    include_Pm_droop = False

    # select whether to use a simple LQR controller 
    use_lqr = True
    # time step horizon for MPC, if applicable
    N = 4

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
        x0[ngen:ngen+1] = -.1 
        x0[ngen+1:ngen+2] = .08

    # set up controller
    if use_lqr:
        K, qp_info = build_lqr(sim_data, x0, N, h, omega_weight, rocof_weight, dPibr_weight, constraint_mode, include_Pm_droop, use_paper_ss)
    else:
        qp, qp_info = build_qp(sim_data, x0, N, h, omega_weight, rocof_weight, dPibr_weight, constraint_mode, include_Pm_droop, use_paper_ss)

    dPmaxs = np.array([ele.dPmax for ele in sim_data['ibr']])
    dPmins = np.array([ele.dPmin for ele in sim_data['ibr']])

    # run simulation
    times = np.arange(0, tf+h, h)
    Nt = len(times)
    results_hist = {}
    states_hist = np.zeros((n, Nt))
    inputs_hist = np.zeros((m, Nt-1))
    states_hist[:,0] = x0
    x_prev = x0

    for t_ind in range(Nt-1):
        t = times[t_ind]
        if use_lqr:
            u = -K @ x_prev
        else:
            results = qp.solve()
            results_hist[t] = results
            u = results.x[:m]
        # apply saturation to u
        u_min = np.linalg.inv(sim_data['Bii'])@(dPmins - sim_data['Big']@x_prev[:ngen])
        u_max = np.linalg.inv(sim_data['Bii'])@(dPmaxs - sim_data['Big']@x_prev[:ngen])
        u_sat = np.minimum(u_max, np.maximum(u, u_min))
        if u != u_sat:
            u = u_sat
        inputs_hist[:,0] = u
        print("t: %.2f" % t)
        print("Gen angles: ", x_prev[:ngen]*180/np.pi)
        print("domega: ", x_prev[ngen:])
        print("IBR angles: ", u*180/np.pi)
        # ideally run with more accurate dynamics
        x_next = DC_dynamics(x_prev, u, t, sim_data, qp_info)
        states_hist[:,t_ind+1] = x_next
        x_prev = np.copy(x_next)
        if not use_lqr:
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

    # calculate the dPs (up to N-1)
    theta_hist = states_hist[:ngen, :Nt-1]
    dP_ibr = (sim_data['Big'] @ theta_hist) + (sim_data['Bii'] @ inputs_hist)

    ibr0 = sim_data['ibr'][0]
    plt.figure(2)
    ibr_P = ibr0.P + dP_ibr[0,:]
    plt.plot(times[:-1], ibr_P) 
    plt.xlabel('Simulation time (sec)')
    plt.ylabel('IBR output power (p.u.)')

    plt.show()


if __name__ == "__main__":
    run_ibr_controller()