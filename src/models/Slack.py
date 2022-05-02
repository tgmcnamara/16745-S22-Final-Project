from __future__ import division
import numpy as np
from models.Buses import Buses
from models.Buses import Buses

class Slack:

    def __init__(self,
                 Bus,
                 Vset,
                 ang,
                 Pinit,
                 Qinit):
        """Initialize slack bus in the power grid.

        Args:
            Bus (int): the bus number corresponding to the slack bus.
            Vset (float): the voltage setpoint that the slack bus must remain fixed at.
            ang (float): the slack bus voltage angle that it remains fixed at.
            Pinit (float): the initial active power that the slack bus is supplying
            Qinit (float): the initial reactive power that the slack bus is supplying
        """
        # You will need to implement the remainder of the __init__ function yourself.
        self.Bus = Bus
        self.Vset = Vset
        self.ang = ang
        self.Pinit = Pinit
        self.Qinit = Qinit
        # initialize
        self.Vr_set = Vset*np.cos(ang*np.pi/180)
        self.Vi_set = Vset*np.sin(ang*np.pi/180)

    def assign_indexes(self, bus):
        """Assign the additional slack bus nodes for a slack bus.
        Returns:
            None
        """
        self.Vr_node = bus[Buses.bus_key_[self.Bus]].Vr_node
        self.Vi_node = bus[Buses.bus_key_[self.Bus]].Vi_node
        self.P_node = Buses._node_index.__next__()
        self.Q_node = Buses._node_index.__next__()