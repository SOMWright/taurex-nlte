import numpy as np


class NLTEOpacityMixin():

    @property
    def vibTemperatureGrid(self):
        return None

    def compute_opacity(self, temperature, pressure, wngrid=None, temperature_vib=None):
        """
        Must return in units of cm2
        """
        raise NotImplementedError

    def opacity(self, temperature, pressure, wngrid=None, temperature_vib=None):

        if wngrid is None:
            wngrid_filter = slice(None)
        else:
            wngrid_filter = np.where((self.wavenumberGrid >= wngrid.min()) & (
                    self.wavenumberGrid <= wngrid.max()))[0]

        orig = self.compute_opacity(temperature, pressure, wngrid_filter, temperature_vib=temperature_vib)

        if wngrid is None or np.array_equal(self.wavenumberGrid.take(wngrid_filter), wngrid):
            return orig
        else:
            return np.interp(wngrid, self.wavenumberGrid[wngrid_filter], orig)
