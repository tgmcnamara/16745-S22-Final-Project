from __future__ import division
from itertools import count
from .Buses import Buses
import numpy as np

class Transformers:
    _ids = count(0)

    def __init__(self,
                 from_bus,
                 to_bus,
                 r,
                 x,
                 status,
                 tr,
                 ang,
                 Gsh_raw,
                 Bsh_raw,
                 rating):
        """Initialize a transformer instance

        Args:
            from_bus (int): the primary or sending end bus of the transformer.
            to_bus (int): the secondary or receiving end bus of the transformer
            r (float): the line resitance of the transformer in
            x (float): the line reactance of the transformer
            status (int): indicates if the transformer is active or not
            tr (float): transformer turns ratio
            ang (float): the phase shift angle of the transformer
            Gsh_raw (float): the shunt conductance of the transformer
            Bsh_raw (float): the shunt admittance of the transformer
            rating (float): the rating in MVA of the transformer
        """
        self.from_bus = from_bus
        self.to_bus = to_bus
        self.r = r
        self.x = x
        self.status = status
        self.tr = tr
        self.ang = ang
        self.theta = ang*np.pi/180.0
        self.Gsh_raw = Gsh_raw
        self.Bsh_raw = Bsh_raw
        self.rating = rating

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

    def stamp_admittance(self, Y):
        ys = self.G_pu + 1j*self.B_pu

        Y[self.from_index, self.from_index] += (ys + 1j*self.Bsh_raw)/self.tr**2
        Y[self.from_index, self.to_index] += -ys*1/(self.tr*np.exp(-1j*self.theta))
        Y[self.to_index, self.from_index] += -ys*1/(self.tr*np.exp(1j*self.theta))
        Y[self.to_index, self.to_index] += (ys + 1j*self.Bsh_raw)