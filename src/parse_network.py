import numpy as np
import json
from itertools import count
from parsers.Data import Bus
from parsers.parser import parse_raw
from models.Buses import Buses

def parse_network(rawfile, jsonfile, constraint_mode, include_Pm_droop):
    # using a parser from our lab group
    # read data from file
    parsed_data = parse_raw(rawfile)
    bus = parsed_data['buses']
    slack = parsed_data['slack']
    generator = parsed_data['generators']
    transformer = parsed_data['xfmrs']
    branch = parsed_data['branches']
    shunt = parsed_data['shunts']
    load = parsed_data['loads']

    with open(jsonfile, 'r') as f:
        extra_data = json.load(f)

    ibr_set = set([ele['bus'] for ele in extra_data['ibr']])

    # sort buses (synch gen buses first, then ibr gen buses, then load buses)
    gen_buses = [ele for ele in bus if ele.Type != 1 and ele.Bus not in ibr_set]
    ibr_buses = [ele for ele in bus if ele.Type != 1 and ele.Bus in ibr_set]
    pq_buses = [ele for ele in bus if ele.Type == 1]

    ng = len(gen_buses)
    ni = len(ibr_buses)

    state_counter = count(0)
    input_counter = count(0)
    synch_gen = []
    ibr = []
    pmax_tot = sum([ele.Pmax for ele in generator])
    for ele in gen_buses:
        ele.assign_nodes()
        # get corresponding gen and assign its nodes
        gen_ele = [g for g in generator if g.Bus == ele.Bus][0]
        gen_data = [gd for gd in extra_data['synch_gen'] if gd['bus'] == ele.Bus][0]
        gen_ele.IBR = False
        gen_ele.inertia = gen_data['inertia']/(60*2*np.pi)
        # an alternative method of estimating inertia
        gen_ele.inertia_alt = gen_ele.Pmax/pmax_tot
        gen_ele.damping = gen_data['damping']
        gen_ele.droop_R = gen_data['droop_R']
        gen_ele.droop_T = gen_data['droop_T']
        gen_ele.assign_indexes_ctrl(state_counter, input_counter, ng, constraint_mode, include_Pm_droop)
        synch_gen.append(gen_ele)
    for ele in ibr_buses:
        ele.assign_nodes()
        ibr_ele = [g for g in generator if g.Bus == ele.Bus][0]
        ibr_ele.IBR = True
        ibr_ele.assign_indexes_ctrl(state_counter, input_counter, ng, constraint_mode, include_Pm_droop)
        ibr.append(ibr_ele)
    for ele in pq_buses:
        ele.assign_nodes()
    n = 2*ng
    if include_Pm_droop:
        n += ng
    if constraint_mode == 2:
        n += ni
    m = input_counter.__next__()

    bus_idx = 0
    for ele in bus:
        Buses.bus_key_[ele.Bus] = bus_idx
        bus_idx += 1

    for disturbance in extra_data['disturbance']:
        if disturbance['type'] == "gen":
            gen_ele = [g for g in generator if g.Bus == disturbance['bus']][0]
            gen_ele.disturbance_t_start = disturbance['tstart']
            gen_ele.disturbance_t_stop = disturbance['tstop']
            gen_ele.disturbance_dP = -disturbance['scale']*gen_ele.P

    # solve dc power flow to get initial P values
    # (just in case they werent the solution values in the .raw file)
    for ele in branch:
        ele.assign_indexes(bus)
    for ele in transformer:
        ele.assign_indexes(bus)
    for ele in generator:
        ele.assign_indexes(bus)
    for ele in load:
        ele.assign_indexes(bus)
    for ele in shunt:
        ele.assign_indexes(bus)
    slack_bus = [ele for ele in bus if ele.Type == 3][0]
    slack_ind = bus[Buses.bus_key_[slack_bus.Bus]].bus_index

    # generate Y matrix
    size_Y = Buses._node_index.__next__()
    Bmat = np.zeros((size_Y, size_Y), dtype=np.float)
    for ele in branch:
        ele.stamp_admittance_dc(Bmat)
    for ele in transformer:
        ele.stamp_admittance_dc(Bmat)

    # generate delta P vector
    Pload = np.zeros(size_Y)
    Pgen = np.zeros(size_Y)
    for ele in generator:
        Pgen[ele.bus_index] += ele.P
    for ele in load:
        Pload[ele.bus_index] += ele.P
    for ele in shunt:
        Pload[ele.bus_index] += ele.G_pu
    dP_full = Pgen - Pload
    # TODO - include Pshift term for transformers    

    # run DC power flow
    # remove slack row/col
    B_dcpf = np.delete(Bmat, slack_ind, axis=0)
    B_dcpf = np.delete(B_dcpf, slack_ind, axis=1)
    dP_dcpf = np.delete(dP_full, slack_ind)
    th_sol_red = np.linalg.solve(B_dcpf, dP_dcpf)

    print("DC Power Flow solutions:")
    th_sol = np.insert(th_sol_red, slack_ind, 0.0) + slack_bus.Va_init*np.pi/180
    P_sol = Bmat @ th_sol
    for ind in range(len(bus)):
        bus_ele = bus[Buses.bus_key_[(ind+1)]]
        th = th_sol[bus_ele.bus_index]
        bus_ele.th_dc_sol = th
        print("Bus %d: %.3f" % (ind+1, th*180/np.pi))
    for ele in generator:
        ele.Pss = P_sol[ele.bus_index] + Pload[ele.bus_index]
        if not ele.IBR:
            ele.Pss += ele.damping # assuming damping value is per-unitized
        ele.dPmax = ele.Pmax - ele.Pss
        ele.dPmin = ele.Pmin - ele.Pss
            
    # apply kron reduction to B matrix
    # so that we get the direct relationship
    # of power flows to gen bus angles 
    # (can bypass calculating load bus angles)
    b0_ind = ibr_buses[0].bus_index
    b1_ind = pq_buses[0].bus_index
    Bgg = Bmat[:b0_ind, :b0_ind]
    Bgi = Bmat[:b0_ind, b0_ind:b1_ind]
    Bgl = Bmat[:b0_ind, b1_ind:]
    Big = Bmat[b0_ind:b1_ind, :b0_ind]
    Bii = Bmat[b0_ind:b1_ind, b0_ind:b1_ind]
    Bil = Bmat[b0_ind:b1_ind, b1_ind:]
    Blg = Bmat[b1_ind:, :b0_ind]
    Bli = Bmat[b1_ind:, b0_ind:b1_ind]
    Bll = Bmat[b1_ind:, b1_ind:]

    Bll_inv = np.linalg.inv(Bll)
    Bgg_hat = Bgg - (Bgl @ Bll_inv @ Blg)
    Bgi_hat = Bgi - (Bgl @ Bll_inv @ Bli)
    Big_hat = Big - (Bil @ Bll_inv @ Blg)
    Bii_hat = Bii - (Bil @ Bll_inv @ Bli)
    B_hat = np.block([[Bgg_hat,Bgi_hat], [Big_hat, Bii_hat]])

    sim_data = {
        'synch_gen': synch_gen,
        'ibr': ibr,
        'n': n,
        'm': m,
        'load': load,
        'Bgg': Bgg_hat,
        'Bgi': Bgi_hat,
        'Big': Big_hat,
        'Bii': Bii_hat,
        'Bhat': B_hat
    }

    return sim_data