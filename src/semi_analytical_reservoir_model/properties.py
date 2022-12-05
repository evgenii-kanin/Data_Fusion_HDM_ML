import numpy as np


class PropertiesClass:
    def __init__(self, *args):
        """
        args:
            0 -- time_array [semi_analytical_reservoir_model], 1 -- flow_rate [m^3 / day],
            2 -- mu [cP], 3 -- b [m^3 /st. m^3], 4 -- ct [1 / atm],
            5 -- k [mD], 6 -- phi, 7 -- p_init [atm], 8 -- rw [m],
            9 -- portion_x, 10 -- portion_y,
            11 -- xe [m], 12 -- ye [m], 13 -- ze [m],
            14 -- xbound, 15 -- ybound,
            16 -- skin
        """

        self.time_array = args[0]
        self.flow_rate = args[1]

        self.xbound, self.ybound = args[14], args[15]
        xe, ye, ze = args[11], args[12], args[13]

        rw = args[8]
        self.skin = args[16]

        if self.skin < 0:
            rw *= np.exp(-self.skin)
            self.skin = 0

        xw, yw = xe * args[9], ye * args[10]
        x, y = xw + rw, yw

        self.xed, self.yed = np.array([xe, ye]) / rw
        self.xwd, self.ywd = np.array([xw, yw]) / rw
        self.xd, self.yd = np.array([x, y]) / rw

        self.mult_t = args[4] * args[6] * args[2] * rw ** 2 / (0.00036 * args[5])
        self.mult_p = 18.42 * args[2] * args[3] / (args[5] * ze)

        self.p_init = args[7]

        self.td_array = self.time_array / self.mult_t
