import numpy as np
from taurex.opacity import HDF5Opacity
from taurex.mpi import allocate_as_shared
from .NLTEOpacityMixin import NLTEOpacityMixin
from .NLTEInterpolationMixin import NLTEInterpolationMixin


class NLTEHDF5Opacity(HDF5Opacity, NLTEInterpolationMixin, NLTEOpacityMixin):

    @classmethod
    def priority(cls):
        # Ensure that Non LTE HDF5 Class is tried first when loading opacities
        return 1

    def _load_hdf_file(self, filename):
        import h5py
        import astropy.units as u
        # Load the pickle file
        self.debug('Loading opacity from {}'.format(filename))

        self._spec_dict = h5py.File(filename, 'r')

        self._wavenumber_grid = self._spec_dict['bin_edges'][:]

        # temperature_units = self._spec_dict['t'].attrs['units']
        # t_conversion = u.Unit(temperature_units).to(u.K).value

        spec_dict_keys = self._spec_dict.keys()

        if 't' in spec_dict_keys:
            # Handle LTE Cross Section HDF5 File
            self._temperature_grid = self._spec_dict['t'][:]  # *t_conversion
            self._vib_temperature_grid = None
        elif 'tv' in spec_dict_keys and 'tr' in spec_dict_keys and 't' not in spec_dict_keys:
            # Handle NLTE Cross Section HDF5 File
            self._temperature_grid = self._spec_dict['tr'][:]  # *t_conversion
            self._vib_temperature_grid = self._spec_dict['tv'][:]  # *t_conversion
            self._min_vib_temperature = self._vib_temperature_grid.min()
            self._max_vib_temperature = self._vib_temperature_grid.max()

        pressure_units = self._spec_dict['p'].attrs['units']
        try:
            p_conversion = u.Unit(pressure_units).to(u.Pa)
        except:
            p_conversion = u.Unit(pressure_units, format="cds").to(u.Pa)

        self._pressure_grid = self._spec_dict['p'][:] * p_conversion

        if self.in_memory:
            self._xsec_grid = allocate_as_shared(self._spec_dict['xsecarr'][...], logger=self)
        else:
            self._xsec_grid = self._spec_dict['xsecarr']

        self._resolution = np.average(np.diff(self._wavenumber_grid))
        self._molecule_name = self._spec_dict['mol_name'][()]

        if isinstance(self._molecule_name, np.ndarray):
            self._molecule_name = self._molecule_name[0]

        try:
            self._molecule_name = self._molecule_name.decode()
        except (UnicodeDecodeError, AttributeError,):
            pass

        from taurex.util.util import ensure_string_utf8

        self._molecule_name = ensure_string_utf8(self._molecule_name)

        self._min_pressure = self._pressure_grid.min()
        self._max_pressure = self._pressure_grid.max()
        self._min_temperature = self._temperature_grid.min()
        self._max_temperature = self._temperature_grid.max()

        if self.in_memory:
            self._spec_dict.close()

    @property
    def vibTemperatureGrid(self):
        return self._vib_temperature_grid
