import numpy as np
import sys
import json
import os
from parse_network import parse_network
from build_QP_matrices import build_qp, update_qp, build_lqr
from true_dynamics import DC_dynamics
from build_observer import observer
from matplotlib import pyplot as plt

sys.path.append("..")

# This is if we aren't using the observer
def get_disturbance(t, sim_data):
    d = np.zeros(len(sim_data['synch_gen']))
    for ele in sim_data['synch_gen']:
        d[ele.state_index] += ele.check_dP(t)
    return d

def run_ibr_controller():
    # parse file and get initial state
    # 0 - 9 bus, 3 gen (1 IBR)
    # This one is very hard to get stable
    # 1 - 38 bus, 10 gen (2 IBR)
    case = 0
    if case == 1:
        rawfile = 'ieee38_dc_sol.raw'
        jsonfile = 'ieee38_mpc_data.json'
    else:
        rawfile = 'case9_dc_sol.raw'
        jsonfile = 'case9_mpc_data.json'
    # uncomment one of these depending on which experiment you want to run
    # settings_file = 'lqr_9bus_settings.json'
    # settings_file = 'mpc10_9bus_settings.json'
    settings_file = 'lqr_9bus_disturbed_settings.json'
    # settings_file = 'mpc10_disturbed_9bus_settings.json'
    # settings_file = 'mpc20_disturbed_9bus_settings.json'
    # settings_file = 'lqr_38bus_settings.json'
    # settings_file = 'mpc10_38bus_settings.json'

    with open(settings_file, 'r') as f:
        settings = json.load(f)
    
    # define simulation parameters
    tf = settings['tf']
    h = settings['h']
    # penalize frequency deviation from 60 Hz
    omega_weight = settings['omega_weight']
    # penalize rate of change of frequency deviation
    rocof_weight = settings['rocof_weight']
    # penalize use of IBRs (can leave at 0)
    dPibr_weight = settings['dPibr_weight']
    
    # model the synch gen's governor droop controller in state space model
    # this hasn't been tested extensively and was not really discussed in the report
    include_Pm_droop = settings['include_Pm_droop']

    # time step horizon for MPC, if applicable
    N = settings['N']

    # constraint mode:
    # 0 - no constraints
    # 1 - upper and lower bounds on dP for IBRs
    # 2 - battery energy relation to dP for IBRs 
    #     doesn't really work yet :(
    constraint_mode = settings['constraint_mode']
    # select whether to use a simple LQR controller 
    use_lqr = settings['use_lqr']
    
    model_disturbance = settings['model_disturbance']
    plot_file_base = os.path.join('plots', settings['save_file_name'])
    
    sim_data = parse_network(rawfile, jsonfile, constraint_mode, include_Pm_droop)
    n = sim_data['n']
    m = sim_data['m']
    nm = n+m
    ngen = len(sim_data['synch_gen'])
    nibr = len(sim_data['ibr'])

    # if testing disturbance response, 
    # we assume we're at steady state before
    # so dtheta and domega are 0 initially
    x0 = np.zeros(n)
    if constraint_mode ==2:
        Ebatt_inits = np.array([ele.Ebatt_init for ele in sim_data['ibr']])
        x0[-nibr:] = Ebatt_inits
    d = np.zeros(ngen)

    # if not testing disturbance, just check
    # ability of controller to drive state back to origin
    if not model_disturbance:
        # some small initial freq deviations
        np.random.seed(2)
        # one option...
        # deviations of ~ -1Hz with some added randomness
        # x0[ngen:2*ngen] = (-1 + 0.1*np.random.randn(ngen))*(2*np.pi) 

        x0[ngen] = (-0.1)*(2*np.pi) 
        x0[ngen+1] = (.1)*(2*np.pi) 

    # set up controller
    if use_lqr:
        K, qp_info = build_lqr(sim_data, x0, N, h, omega_weight, rocof_weight, dPibr_weight, constraint_mode, include_Pm_droop)
    else:
        qp, qp_info = build_qp(sim_data, x0, d, N, h, omega_weight, rocof_weight, dPibr_weight, constraint_mode, include_Pm_droop)

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

    # Turn on for debugging
    # Just makes sure that MPC output dynamics are correct:
    # A x[k] + B u[k] + Bd d[k] - x[k+1] = 0 
    check_dynamics_constraints = False

    for t_ind in range(Nt-1):
        t = times[t_ind]
        print("t: %.2f" % t)
        if use_lqr:
            u = -K @ x_prev
        else:
            results = qp.solve()
            results_hist[t] = results
            u = results.x[:m]
            # double check that dynamics constaints are correct
            if check_dynamics_constraints:
                z = results.x.reshape(N,nm)
                resid1 = (qp_info.A @ x_prev) + (qp_info.Bu @ u) + (qp_info.Bd @ d) - z[0,m:]
                print("1 step ahead dyn residual:")
                print(resid1)
                for i in range(1,N):
                    u_test = z[i,:m]
                    x_test = z[i-1,m:]
                    x_pred = z[i,m:]
                    resid = (qp_info.A @ x_test) + (qp_info.Bu @ u_test) - x_pred
                    print("%d steps ahead dyn residual:" % (i+1))
                    print(resid)
        # apply saturation to u
        u_sat = np.minimum(dPmaxs, np.maximum(u, dPmins))
        if False:#np.amax(np.abs(u - u_sat)) > 1e-4:
            u = u_sat
            
        inputs_hist[:,t_ind] = u
        x_next = DC_dynamics(x_prev, u, d, t, qp_info)
        states_hist[:,t_ind+1] = x_next
        x_prev = np.copy(x_next)
        if model_disturbance:
            d = get_disturbance(times[t_ind+1], sim_data)
        if not use_lqr:
            update_qp(qp, qp_info, sim_data, x_prev, d)
    
    # inspect and plot some results
    rad_to_hz = 1/(2*np.pi)
    plt.style.use('seaborn-darkgrid')
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'font.weight': 'bold'})
    fig, axs = plt.subplots(2,1)
    fig.set_size_inches(7, 8)
    for i in range(ngen):
        freq_g = states_hist[sim_data['synch_gen'][i].domega_index,:]*rad_to_hz + 60
        axs[0].plot(times, freq_g)
    axs[0].set_xlabel('Simulation time (sec)')
    axs[0].set_ylabel('Generator frequency (Hz)')
    axs[0].legend(["Gen 1", "Gen 2"])
    # fig_path = plot_file_base + '_gen_freq_dev.svg'
    # plt.savefig(fig_path, format='svg')

    # calculate the dPs (up to N-1)
    for i in range(nibr):
        axs[1].plot(times[:-1], inputs_hist[i,:]) 
    axs[1].set_xlabel('Simulation time (sec)')
    axs[1].set_ylabel('IBR output delta P (p.u.)')
    fig_path = plot_file_base + '-plots.png'
    fig.savefig(fig_path, format='png')
    
    if constraint_mode == 2:
        plt.figure(2)
        for i in range(nibr):
            ibr_ele = sim_data['ibr'][i]
            Ebatt_i = states_hist[ibr_ele.Ebatt_index,:]/ibr_ele.Ebatt_max * 100
            plt.plot(times, Ebatt_i)
            plt.xlabel('Simulation time (sec)')
            plt.ylabel('IBR Battery State of Charge (%)')
    
    
    plt.show()
    test = 2 + 2

if __name__ == "__main__":
    run_ibr_controller()