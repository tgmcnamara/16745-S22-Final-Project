from __future__ import division
from itertools import count
from .Buses import Buses
from .global_vars import global_vars

class Generators:
    _ids = count(0)
    RemoteBusGens = dict()
    RemoteBusRMPCT = dict()
    gen_bus_key_ = {}
    total_P = 0

    def __init__(self,
                 Bus,
                 P,
                 Vset,
                 Qmax,
                 Qmin,
                 Pmax,
                 Pmin,
                 Qinit,
                 RemoteBus,
                 RMPCT,
                 gen_type):
        """Initialize an instance of a generator in the power grid.

        Args:
            Bus (int): the bus number where the generator is located.
            P (float): the current amount of active power the generator is providing.
            Vset (float): the voltage setpoint that the generator must remain fixed at.
            Qmax (float): maximum reactive power
            Qmin (float): minimum reactive power
            Pmax (float): maximum active power
            Pmin (float): minimum active power
            Qinit (float): the initial amount of reactive power that the generator is supplying or absorbing.
            RemoteBus (int): the remote bus that the generator is controlling
            RMPCT (float): the percent of total MVAR required to hand the voltage at the controlled bus
            gen_type (str): the type of generator
        """

        self.Bus = Bus
        self.P_MW = P
        self.Vset = Vset
        self.Qmax_MVAR = Qmax
        self.Qmin_MVAR = Qmin
        self.Pmax_MW = Pmax
        self.Pmin_MW = Pmin
        self.Qinit_MVAR = Qinit
        self.RemoteBus = RemoteBus
        self.RMPCT = RMPCT
        self.gen_type = gen_type
        # convert P/Q to pu
        self.P = P/global_vars.base_MVA
        self.Vset = Vset
        self.Qmax = Qmax/global_vars.base_MVA
        self.Qmin = Qmin/global_vars.base_MVA
        self.Pmax = Pmax/global_vars.base_MVA
        self.Pmin = Pmin/global_vars.base_MVA
        if self.Bus == 3:
            self.Pmax = 1.0
            self.Pmin = 0.7

        self.id = self._ids.__next__()

        # variables if its a synch gen
        self.inertia = None
        self.damping = None
        self.droop_R = None
        self.droop_T = None
        self.droop_Pss = None
        
        self.ddelta_index = None # int - index in state vector
        self.domega_index = None # int - index in state vector
        self.dPmech_index = None # int - index in state vector, if applicable

        self.disturbance_t_start = None
        self.disturbance_t_stop = None
        self.disturbance_dP = None

        # variables if it is an IBR
        self.IBR = None # bool
        self.Ebatt_index = None # int - index in state vector, if applicable
        self.input_index = None # int - index in U
        # capacity of 36 p.u seconds --> 1 megawatt hour of storage, 
        # would coset around $400k according to NREL report 
        # (https://www.nrel.gov/docs/fy21osti/79236.pdf)
        self.Ebatt_max = 36.0 
        self.Ebatt_init = 30.0
        self.Ebatt_min = 0.0

    def assign_indexes(self, bus):
        self.bus_index = bus[Buses.bus_key_[self.Bus]].bus_index

    def assign_indexes_ctrl(self, state_counter, input_counter, ng, constraint_mode, include_Pm_droop=False):
        # index order should be:
        # -all synch gen angles
        # -all synch gen frequencies
        # -all gen Pmech to track droop LPF (if applicable)
        # -IBR battery state of charge (if applicable)
        if self.IBR:
            self.input_index = input_counter.__next__()
            if constraint_mode == 2:
                self.state_index = state_counter.__next__()
                self.Ebatt_index = self.state_index + 3*ng
        else:
            self.state_index = state_counter.__next__() 
            self.dtheta_index = self.state_index
            self.domega_index = self.state_index + ng
            if include_Pm_droop:
                self.dPmech_index = self.state_index + 2*ng

    def check_dP(self, t):
        # include droop dP here?
        if (self.disturbance_t_start != None) and self.disturbance_t_start <= t <= self.disturbance_t_stop:
            return self.disturbance_dP
        else:
            return 0.0