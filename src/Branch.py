from IBR import IBR
from SynchGen import SynchGen

class Branch:
    def __init__(self,
                 from_bus,
                 to_bus,
                 r,
                 x):
        self.from_bus = from_bus
        self.to_bus = to_bus
        self.r = r
        self.x = x
        self.g = r/(r**2+x**2)
        self.b = -x/(r**2+x**2)
        
        # Should really check to make sure there isn't a SynchGen
        # and an IBR on the same bus
        self.from_bus_gen = self.from_bus in SynchGen.gen_key_
        self.to_bus_gen = self.from_bus in SynchGen.gen_key_
        self.from_bus_ibr = self.from_bus in IBR.ibr_key_
        self.to_bus_ibr = self.from_bus in IBR.ibr_key_


    def assign_indexes(self, gens, ibrs):
        if self.from_bus_gen:
            self.from_bus_index = gens[SynchGen.gen_key_[self.from_bus]].ddelta_index
        elif self.from_bus_ibr:
            self.from_bus_index = gens[IBR.ibr_key_[self.from_bus]].ctrl_angle_index
        if self.to_bus_gen:
            self.to_bus_index = ibrs[SynchGen.gen_key_[self.to_bus]].ddelta_index
        elif self.from_bus_ibr:
            self.to_bus_index = ibrs[IBR.ibr_key_[self.to_bus]].ctrl_angle_index