from __future__ import division
from itertools import count
from .Buses import Buses

class Branches:
    _ids = count(0)

    def __init__(self,
                 from_bus,
                 to_bus,
                 r,
                 x,
                 b,
                 status,
                 rateA,
                 rateB,
                 rateC):
        """Initialize a branch in the power grid.

        Args:
            from_bus (int): the bus number at the sending end of the branch.
            to_bus (int): the bus number at the receiving end of the branch.
            r (float): the branch resistance
            x (float): the branch reactance
            b (float): the branch susceptance
            status (bool): indicates if the branch is online or offline
            rateA (float): The 1st rating of the line.
            rateB (float): The 2nd rating of the line
            rateC (float): The 3rd rating of the line.
        """
        self.from_bus = from_bus
        self.to_bus = to_bus
        self.r = r
        self.x = x
        self.b = b
        self.status = bool(status)
        self.rateA = rateA
        self.rateB = rateB
        self.rateC = rateC

        # Set minimum x:
        if abs(self.x) < 1e-6:
            if self.x < 0:
                self.x = -1e-6
            else:
                self.x = 1e-6

        # convert to G and B
        self.G_pu = self.r/(self.r**2+self.x**2)
        self.B_pu= -self.x/(self.r**2+self.x**2)

        self.id = self._ids.__next__()

    def assign_indexes(self, bus):
        self.from_index = bus[Buses.bus_key_[self.from_bus]].bus_index
        self.to_index = bus[Buses.bus_key_[self.to_bus]].bus_index
        # self.Vr_from_node = bus[Buses.bus_key_[self.from_bus]].Vr_node
        # self.Vi_from_node = bus[Buses.bus_key_[self.from_bus]].Vi_node
        # self.Vr_to_node = bus[Buses.bus_key_[self.to_bus]].Vr_node
        # self.Vi_to_node = bus[Buses.bus_key_[self.to_bus]].Vi_node

    def stamp_admittance(self, Y):
        Y[self.from_index, self.from_index] += self.G_pu + 1j*self.B_pu
        Y[self.from_index, self.to_index] += -(self.G_pu + 1j*self.B_pu)
        Y[self.to_index, self.from_index] += -(self.G_pu + 1j*self.B_pu)
        Y[self.to_index, self.to_index] += (self.G_pu + 1j*self.B_pu)