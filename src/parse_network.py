import numpy as np
import json
from itertools import count
from parsers.Data import Bus
from parsers.parser import parse_raw
from models.Buses import Buses

def parse_network(rawfile, jsonfile):
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

    state_counter = count(0)
    input_counter = count(0)
    synch_gen = []
    ibr = []
    for ele in gen_buses:
        ele.assign_nodes()
        # get corresponding gen and assign its notes
        # this is a crap way to do this
        gen_ele = [g for g in generator if g.Bus == ele.Bus][0]
        gen_data = [gd for gd in extra_data['synch_gen'] if gd['bus'] == ele.Bus][0]
        gen_ele.IBR = False
        gen_ele.inertia = gen_data['inertia']*2/(120*np.pi)
        gen_ele.damping = gen_data['damping']
        gen_ele.droop = gen_data['droop']
        gen_ele.assign_indexes_MPC(state_counter, input_counter, len(gen_buses))
        synch_gen.append(gen_ele)
    for ele in ibr_buses:
        ele.assign_nodes()
        ibr_ele = [g for g in generator if g.Bus == ele.Bus][0]
        ibr_ele.IBR = True
        ibr_ele.assign_indexes_MPC(state_counter, input_counter, len(gen_buses))
        ibr.append(ibr_ele)
    for ele in pq_buses:
        ele.assign_nodes()

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


    for ele in branch:
        ele.assign_indexes(bus)
    for ele in transformer:
        ele.assign_indexes(bus)
    for ele in generator:
        ele.assign_indexes(bus)
    for ele in load:
        ele.assign_indexes(bus)
    slack_bus = [ele for ele in bus if ele.Type == 3][0]
    slack_ind = bus[Buses.bus_key_[slack_bus.Bus]].bus_index

    # generate Y matrix
    size_Y = Buses._node_index.__next__()
    Yfull = np.zeros((size_Y, size_Y), dtype=np.complex128)
    for ele in branch:
        ele.stamp_admittance(Yfull)
    for ele in transformer:
        ele.stamp_admittance(Yfull)
    # isolate the complex admittance part
    Bmat = np.imag(Yfull)

    zr = []
    zc = []
    for ind in range(Bmat.shape[0]):
        if np.count_nonzero(Bmat[ind,:]) == 0:
            zr.append(ind)
        if np.count_nonzero(Bmat[:,ind]) == 0:
            zc.append(ind)

    # generate delta P vector
    dP_full = np.zeros(size_Y)
    for ele in generator:
        dP_full[ele.bus_index] += ele.P
    for ele in generator:
        dP_full[ele.bus_index] -= ele.P

    # run DC power flow
    # remove slack row/col
    B_dcpf = np.delete(Bmat, slack_ind, axis=0)
    B_dcpf = np.delete(B_dcpf, slack_ind, axis=1)
    dP_dcpf = np.delete(dP_full, slack_ind)
    dcpf_sol = np.linalg.solve(B_dcpf, dP_dcpf)
    
    # apply kron reduction to B matrix
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

    sim_data = {
        'synch_gen': synch_gen,
        'ibr': ibr,
        'load': load,
        'Bgg': Bgg_hat,
        'Bgi': Bgi_hat,
        'Big': Big_hat,
        'Bii': Bii_hat,
    }

    return sim_data