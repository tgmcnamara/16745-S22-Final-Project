import numpy as np
import sys
from parse_network import parse_network
from build_QP_matrices import build_qp, update_qp, build_lqr
from true_dynamics import DC_dynamics
from matplotlib import pyplot as plt

sys.path.append("..")

# This is if we aren't using the observer
def get_disturbance(t, sim_data):
    d = np.zeros(len(sim_data['synch_gen']))
    for ele in sim_data['synch_gen']:
        d[ele.state_index] += ele.check_dP(t)
    return d

def run_ibr_controller():
    # define simulation parameters
    tf = 20
    h = 0.05
    # penalize frequency deviation from 60 Hz
    omega_weight = 10.0
    # penalize rate of change of frequency deviation
    rocof_weight = 0.01
    # penalize use of IBRs (can leave at 0)
    dPibr_weight = 0.001
    
    # model the synch gen's governor droop controller in state space model
    include_Pm_droop = False

    # time step horizon for MPC, if applicable
    N = 10

    # constraint mode:
    # 0 - no constraints
    # 1 - upper and lower bounds on dP for IBRs
    # 2 - battery energy relation to dP for IBRs
    constraint_mode = 1
    # select whether to use a simple LQR controller 
    use_lqr = False
    
    # parse file and get initial state
    # 0 - 9 bus, 3 gen (1 IBR)
    # 1 - 38 bus, 10 gen (2 IBR)
    case = 0

    if case == 1:
        rawfile = 'ieee38_dc_sol.raw'
        jsonfile = 'ieee38_mpc_data.json'
    else:
        rawfile = 'case9_dc_sol.raw'
        jsonfile = 'case9_mpc_data.json'

    sim_data = parse_network(rawfile, jsonfile, constraint_mode, include_Pm_droop)
    n = sim_data['n']
    m = sim_data['m']
    nm = n+m
    ngen = len(sim_data['synch_gen'])
    nibr = len(sim_data['ibr'])

    # if testing disturbance response, 
    # we assume we're at steady state before
    # so dtheta and domega are 0 initially
    model_disturbance = True
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
        # deviations of ~ -1Hz with some added randomness
        x0[ngen:2*ngen] = (-1 + 0.1*np.random.randn(ngen))*(2*np.pi) 

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
        if np.amax(np.abs(u - u_sat)) > 1e-4:
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
    plt.figure(1)
    for i in range(ngen):
        freq_g = states_hist[sim_data['synch_gen'][i].domega_index,:]*rad_to_hz + 60
        plt.plot(times, freq_g)
    plt.xlabel('Simulation time (sec)')
    plt.ylabel('Generator frequency (Hz)')
    # plt.legend(["Gen 1", "Gen 2"])
    plt.savefig('gen_freq_dev.svg', format='svg')

    # calculate the dPs (up to N-1)
    plt.figure(2)
    for i in range(nibr):
        plt.plot(times[:-1], inputs_hist[i,:]) 
    plt.xlabel('Simulation time (sec)')
    plt.ylabel('IBR output delta P (p.u.)')
    plt.savefig('ibr_deltaPs.svg', format='svg')
    
    if constraint_mode == 2:
        plt.figure(3)
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