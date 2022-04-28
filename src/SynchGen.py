from itertools import count

class SynchGen:
    gen_key_ = {}
    _gen_id_counter = count(0)
    _gen_index_counter = count(0)

    def __init__(self,
                 bus,
                 inertia,
                 damping,
                 P0,
                 Rdroop,
                 disturbance_t_start,
                 disturbance_t_stop,
                 disturbance_dP):
        self.bus = bus
        self.inertia = inertia
        self.damping = damping
        self.P0 = P0
        self.Rdroop = Rdroop
        
        self.id = SynchGen._gen_id_counter.__next__()
        SynchGen.gen_key_[self.bus] = self.id

        self.domega_index = SynchGen._gen_index_counter.__next__()
        self.ddelta_index = SynchGen._gen_index_counter.__next__()

        self.disturbance_t_start = disturbance_t_start
        self.disturbance_t_stop = disturbance_t_stop
        self.disturbance_dP = disturbance_dP

    def check_dP(self, t):
        # include droop dP here?
        if self.disturbance_t_start <= t <= self.disturbance_t_stop:
            return self.disturbance_dP
        else:
            return 0.0
