from __future__ import division
from itertools import count
from .Buses import Buses
from .global_vars import global_vars

class Shunts:
    _ids = count(0)

    def __init__(self,
                 Bus,
                 G_MW,
                 B_MVAR,
                 shunt_type,
                 Vhi,
                 Vlo,
                 Bmax,
                 Bmin,
                 Binit,
                 controlBus,
                 flag_control_shunt_bus=False,
                 Nsteps=[0],
                 Bsteps=[0]):

        """ Initialize a shunt in the power grid.
        Args:
            Bus (int): the bus where the shunt is located
            G_MW (float): the active component of the shunt admittance as MW per unit voltage
            B_MVAR (float): reactive component of the shunt admittance as  MVar per unit voltage
            shunt_type (int): the shunt control mode, if switched shunt
            Vhi (float): if switched shunt, the upper voltage limit
            Vlo (float): if switched shunt, the lower voltage limit
            Bmax (float): the maximum shunt susceptance possible if it is a switched shunt
            Bmin (float): the minimum shunt susceptance possible if it is a switched shunt
            Binit (float): the initial switched shunt susceptance
            controlBus (int): the bus that the shunt controls if applicable
            flag_control_shunt_bus (bool): flag that indicates if the shunt should be controlling another bus
            Nsteps (list): the number of steps by which the switched shunt should adjust itself
            Bstep (list): the admittance increase for each step in Nstep as MVar at unity voltage
        """
        self.Bus = Bus
        self.G_MW = G_MW
        self.B_MVAR = B_MVAR
        self.shunt_type = shunt_type
        self.Vhi = Vhi
        self.Vlo = Vlo
        self.Bmax = Bmax
        self.Bmin = Bmin
        self.Binit = Binit
        self.controlBus = controlBus
        self.flag_control_shunt_bus = flag_control_shunt_bus
        self.Nsteps = Nsteps 
        self.Bsteps= Bsteps
        self.id = self._ids.__next__()
        self.G_pu = G_MW/global_vars.base_MVA
        self.B_pu = B_MVAR/global_vars.base_MVA

    def assign_indexes(self, bus):
        self.bus_index = bus[Buses.bus_key_[self.Bus]].bus_index
