from itertools import count

class IBR:
    ibr_key_ = {}
    _ibr_counter = count(0)

    def __init__(self,
                 bus,
                 Pmax,
                 Pmin,
                 Etot,
                 angle_init):
        self.bus = bus
        self.Pmax = Pmax
        self.Pmin = Pmin
        self.Etot = Etot
        self.angle_init = angle_init
        self.id = IBR._ibr_counter.__next__()
        IBR.ibr_key_[self.bus] = self.id
        self.ctrl_angle_index = self.id
        
