import numpy as np
from taurex.opacity import HDF5Opacity
from taurex.mpi import allocate_as_shared
from taurex.util.math import intepr_bilin, interp_exp_and_lin, interp_lin_only, interp_exp_only
from ..util.nltemath import interp_trilin


class NLTEHDF5Opacity(HDF5Opacity):

    @classmethod
    def priority(cls):
        # Ensure that Non LTE HDF5 Class is tried first when loading opacities
        return 0

    @classmethod
    def discover(cls):
        import os
        import glob
        import pathlib
        from taurex.cache import GlobalCache
        from taurex.util.util import sanitize_molecule_string

        path = GlobalCache()['xsec_path']
        if path is None:
            return []
        path = [os.path.join(path, '*.nlteh5'), os.path.join(path, '*.nltehdf5')]
        file_list = [f for glist in path for f in glob.glob(glist)]

        discovery = []

        interp = GlobalCache()['xsec_interpolation'] or 'linear'
        mem = GlobalCache()['xsec_in_memory'] or True

        for f in file_list:
            op = NLTEHDF5Opacity(f, interpolation_mode='linear', in_memory=False)
            mol_name = op.moleculeName
            discovery.append((mol_name, [f, interp, mem]))
            # op._spec_dict.close()
            del op

        return discovery

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

    @property
    def vibTemperatureBounds(self):
        if self.vibTemperatureGrid is not None:
            return self.vibTemperatureGrid.min(), self.vibTemperatureGrid.max()
        else:
            return self.temperatureBounds

    def find_closest_index(self, T, P, Tv=None):
        from taurex.util.util import find_closest_pair
        # t_min = self.temperatureGrid.searchsorted(T, side='right')-1
        # t_min = max(0, t_min)
        # t_max = t_min+1
        # t_max = min(len(self.temperatureGrid)-1, t_max)

        # p_min = self.pressureGrid.searchsorted(P, side='right')-1
        # p_min = max(0, p_min)
        # p_max = p_min+1
        # p_max = min(len(self.pressureGrid)-1, p_max)

        t_min, t_max = find_closest_pair(self.temperatureGrid, T)
        p_min, p_max = find_closest_pair(self.logPressure, P)

        if Tv is not None:
            tv_min, tv_max = find_closest_pair(self.vibTemperatureGrid, Tv)
            return t_min, t_max, tv_min, tv_max, p_min, p_max
        else:
            return t_min, t_max, p_min, p_max

    def interp_pressure_only(self, P, p_idx_min, p_idx_max, T, filt, Tv=None):
        Pmax = self.logPressure[p_idx_max]
        Pmin = self.logPressure[p_idx_min]
        if self.vibTemperatureGrid is not None:
            fx0 = self.xsecGrid[p_idx_min, T, Tv, filt]
            fx1 = self.xsecGrid[p_idx_max, T, Tv, filt]
        else:
            fx0 = self.xsecGrid[p_idx_min, T, filt]
            fx1 = self.xsecGrid[p_idx_max, T, filt]

        return interp_lin_only(fx0, fx1, P, Pmin, Pmax)

    def interp_one_temp(self, T, Tmin, Tmax, fx0, fx1):
        if self._interp_mode == 'linear':
            return interp_lin_only(fx0, fx1, T, Tmin, Tmax)
        elif self._interp_mode == 'exp':
            return interp_exp_only(fx0, fx1, T, Tmin, Tmax)
        else:
            raise ValueError(
                'Unknown interpolation mode {}'.format(self._interp_mode))

    def interp_two_temps(self, Tr, tr_idx_min, tr_idx_max, Tv, tv_idx_min, tv_idx_max, P, wavegrid_filter):
        q_11 = self.xsecGrid[P, tr_idx_min, tv_idx_min][wavegrid_filter].ravel()
        q_12 = self.xsecGrid[P, tr_idx_max, tv_idx_min][wavegrid_filter].ravel()
        q_21 = self.xsecGrid[P, tr_idx_min, tv_idx_max][wavegrid_filter].ravel()
        q_22 = self.xsecGrid[P, tr_idx_max, tv_idx_max][wavegrid_filter].ravel()

        Trmax = self.temperatureGrid[tr_idx_max]
        Trmin = self.temperatureGrid[tr_idx_min]
        Tvmax = self.vibTemperatureGrid[tv_idx_max]
        Tvmin = self.vibTemperatureGrid[tv_idx_min]

        if self._interp_mode == 'linear':
            return intepr_bilin(q_11, q_12, q_21, q_22, Tr, Trmin, Trmax, Tv, Tvmin, Tvmax)
        elif self._interp_mode == 'exp':
            return interp_exp_and_lin(q_11, q_12, q_21, q_22, Tr, Trmin, Trmax, Tv, Tvmin, Tvmax)
        else:
            raise ValueError(
                'Unknown interpolation mode {}'.format(self._interp_mode))

    def interp_pressure_vib_temp(self, Tr, Tv, tv_idx_min, tv_idx_max, P, p_idx_min, p_idx_max, wavegrid_filter):
        q_11 = self.xsecGrid[p_idx_min, Tr, tv_idx_min][wavegrid_filter].ravel()
        q_12 = self.xsecGrid[p_idx_max, Tr, tv_idx_min][wavegrid_filter].ravel()
        q_21 = self.xsecGrid[p_idx_min, Tr, tv_idx_max][wavegrid_filter].ravel()
        q_22 = self.xsecGrid[p_idx_max, Tr, tv_idx_max][wavegrid_filter].ravel()

        Tvmax = self.vibTemperatureGrid[tv_idx_max]
        Tvmin = self.vibTemperatureGrid[tv_idx_min]
        Pmax = self.logPressure[p_idx_max]
        Pmin = self.logPressure[p_idx_min]

        if self._interp_mode == 'linear':
            return intepr_bilin(q_11, q_12, q_21, q_22, P, Pmin, Pmax, Tv, Tvmin, Tvmax)
        elif self._interp_mode == 'exp':
            return interp_exp_and_lin(q_11, q_12, q_21, q_22, P, Pmin, Pmax, Tv, Tvmin, Tvmax)
        else:
            raise ValueError(
                'Unknown interpolation mode {}'.format(self._interp_mode))

    def interp_pressure_rot_temp(self, Tv, Tr, tr_idx_min, tr_idx_max, P, p_idx_min, p_idx_max, wavegrid_filter):
        q_11 = self.xsecGrid[p_idx_min, tr_idx_min, Tv][wavegrid_filter].ravel()
        q_12 = self.xsecGrid[p_idx_max, tr_idx_min, Tv][wavegrid_filter].ravel()
        q_21 = self.xsecGrid[p_idx_min, tr_idx_min, Tv][wavegrid_filter].ravel()
        q_22 = self.xsecGrid[p_idx_max, tr_idx_max, Tv][wavegrid_filter].ravel()

        Trmax = self.vibTemperatureGrid[tr_idx_max]
        Trmin = self.vibTemperatureGrid[tr_idx_min]
        Pmax = self.logPressure[p_idx_max]
        Pmin = self.logPressure[p_idx_min]

        if self._interp_mode == 'linear':
            return intepr_bilin(q_11, q_12, q_21, q_22, P, Pmin, Pmax, Tr, Trmin, Trmax)
        elif self._interp_mode == 'exp':
            return interp_exp_and_lin(q_11, q_12, q_21, q_22, P, Pmin, Pmax, Tr, Trmin, Trmax)
        else:
            raise ValueError(
                'Unknown interpolation mode {}'.format(self._interp_mode))

    def interp_rot_temp_only(self, Tr, tr_idx_min, tr_idx_max, Tv, P, filt):
        Tmax = self.temperatureGrid[tr_idx_max]
        Tmin = self.temperatureGrid[tr_idx_min]
        fx0 = self.xsecGrid[P, tr_idx_min, Tv, filt]
        fx1 = self.xsecGrid[P, tr_idx_max, Tv, filt]

        return self.interp_one_temp(Tr, Tmin, Tmax, fx0, fx1)

    def interp_vib_temp_only(self, Tv, tv_idx_min, tv_idx_max, Tr, P, filt):
        Tmax = self.vibTemperatureGrid[tv_idx_max]
        Tmin = self.vibTemperatureGrid[tv_idx_min]
        fx0 = self.xsecGrid[P, Tr, tv_idx_min, filt]
        fx1 = self.xsecGrid[P, Tr, tv_idx_max, filt]

        return self.interp_one_temp(Tv, Tmin, Tmax, fx0, fx1)

    def interp_bilinear_grid(self, T, P, t_idx_min, t_idx_max, p_idx_min,
                             p_idx_max, wngrid_filter=None):

        self.debug('Interpolating %s %s %s %s %s %s', T, P,
                   t_idx_min, t_idx_max, p_idx_min, p_idx_max)

        if p_idx_max == 0 and t_idx_max == 0:
            return np.zeros_like(self.xsecGrid[0, 0, wngrid_filter]).ravel()

        min_pressure, max_pressure = self.pressureBounds
        min_temperature, max_temperature = self.temperatureBounds

        check_pressure_max = P >= max_pressure
        check_temperature_max = T >= max_temperature

        check_pressure_min = P < min_pressure
        check_temperature_min = T < min_temperature

        self.debug('Check pressure min/max %s/%s',
                   check_pressure_min, check_pressure_max)
        self.debug('Check temeprature min/max %s/%s',
                   check_temperature_min, check_temperature_max)
        # Are we both max?
        if check_pressure_max and check_temperature_max:
            self.debug('Maximum Temperature pressure reached. Using last')
            return self.xsecGrid[-1, -1, wngrid_filter].ravel()

        if check_pressure_min and check_temperature_min:
            return np.zeros_like(self.xsecGrid[0, 0, wngrid_filter]).ravel()

        # Max pressure
        if check_pressure_max:
            self.debug('Max pressure reached. Interpolating temperature only')
            return self.interp_temp_only(T, t_idx_min, t_idx_max, -1, wngrid_filter)

        # Max temperature
        if check_temperature_max:
            self.debug('Max temperature reached. Interpolating pressure only')
            return self.interp_pressure_only(P, p_idx_min, p_idx_max, -1, wngrid_filter)

        if check_pressure_min:
            self.debug('Min pressure reached. Interpolating temperature only')
            return self.interp_temp_only(T, t_idx_min, t_idx_max, 0, wngrid_filter).ravel()

        if check_temperature_min:
            self.debug('Min temperature reached. Interpolating pressure only')
            return self.interp_pressure_only(P, p_idx_min, p_idx_max, 0, wngrid_filter).ravel()

        q_11 = self.xsecGrid[p_idx_min, t_idx_min][wngrid_filter].ravel()
        q_12 = self.xsecGrid[p_idx_min, t_idx_max][wngrid_filter].ravel()
        q_21 = self.xsecGrid[p_idx_max, t_idx_min][wngrid_filter].ravel()
        q_22 = self.xsecGrid[p_idx_max, t_idx_max][wngrid_filter].ravel()

        Tmax = self.temperatureGrid[t_idx_max]
        Tmin = self.temperatureGrid[t_idx_min]
        Pmax = self.logPressure[p_idx_max]
        Pmin = self.logPressure[p_idx_min]

        if self._interp_mode == 'linear':
            return intepr_bilin(q_11, q_12, q_21, q_22, T, Tmin, Tmax, P, Pmin, Pmax)
        elif self._interp_mode == 'exp':
            return interp_exp_and_lin(q_11, q_12, q_21, q_22, T, Tmin, Tmax, P, Pmin, Pmax)
        else:
            raise ValueError(
                'Unknown interpolation mode {}'.format(self._interp_mode))

    def interp_nonlte(self, Tr, Tv, P, tr_idx_min, tr_idx_max, tv_idx_min, tv_idx_max, p_idx_min, p_idx_max,
                      wngrid_filter=None):

        self.debug('Interpolating %s %s %s %s %s %s %s %s %s', Tr, Tv, P,
                   tr_idx_min, tr_idx_max, tv_idx_min, tv_idx_max, p_idx_min, p_idx_max)

        if p_idx_max == 0 and tr_idx_max == 0 and tv_idx_max == 0:
            return np.zeros_like(self.xsecGrid[0, 0, 0, wngrid_filter]).ravel()

        min_pressure, max_pressure = self.pressureBounds
        min_temperature, max_temperature = self.temperatureBounds
        min_vib_temperature, max_vib_temperature = self.vibTemperatureBounds

        check_pressure_max = P >= max_pressure
        check_rot_temperature_max = Tr >= max_temperature
        check_vib_temperature_max = Tv >= max_vib_temperature

        check_pressure_min = P < min_pressure
        check_rot_temperature_min = Tr < min_temperature
        check_vib_temperature_min = Tv < min_vib_temperature

        self.debug('Check pressure min/max %s/%s',
                   check_pressure_min, check_pressure_max)
        self.debug('Check rotational temperature min/max %s/%s',
                   check_rot_temperature_min, check_rot_temperature_max)
        self.debug('Check vibrational temperature min/max %s/%s',
                   check_vib_temperature_min, check_vib_temperature_max)
        # Are we both max?
        if check_pressure_max and check_rot_temperature_max and check_vib_temperature_max:
            self.debug('Maximum Pressure and Temperatures reached. Using last')
            return self.xsecGrid[-1, -1, -1, wngrid_filter].ravel()

        if check_pressure_min and check_rot_temperature_min and check_vib_temperature_min:
            return self.xsecGrid[0, 0, 0, wngrid_filter].ravel()

        if check_pressure_min and check_rot_temperature_min and check_vib_temperature_max:
            return self.xsecGrid[0, 0, -1, wngrid_filter].ravel()

        if check_pressure_min and check_rot_temperature_max and check_vib_temperature_max:
            return self.xsecGrid[0, -1, -1, wngrid_filter].ravel()

        if check_pressure_min and check_rot_temperature_max and check_vib_temperature_min:
            return self.xsecGrid[0, -1, 0, wngrid_filter].ravel()

        if check_pressure_max and check_rot_temperature_max and check_vib_temperature_min:
            return self.xsecGrid[-1, -1, 0, wngrid_filter].ravel()

        if check_pressure_max and check_rot_temperature_min and check_vib_temperature_max:
            return self.xsecGrid[-1, 0, -1, wngrid_filter].ravel()

        if check_pressure_max and check_rot_temperature_min and check_vib_temperature_min:
            return self.xsecGrid[-1, 0, 0, wngrid_filter].ravel()

        if check_pressure_max and check_rot_temperature_max:
            self.debug('Max pressure and rotational temperature reached. Interpolating vibrational temperature only')
            return self.interp_vib_temp_only(Tv, tv_idx_min, tv_idx_max, -1, -1, wngrid_filter).ravel()

        if check_pressure_max and check_vib_temperature_max:
            self.debug('Max pressure and vibrational temperature reached. Interpolating rotational temperature only')
            return self.interp_rot_temp_only(Tr, tr_idx_min, tr_idx_max, -1, -1, wngrid_filter).ravel()

        # Max temperature
        if check_vib_temperature_max and check_rot_temperature_max:
            self.debug('Max temperature reached on both accounts. Interpolating pressure only')
            return self.interp_pressure_only(P, p_idx_min, p_idx_max, -1, wngrid_filter, Tv=-1).ravel()

        if check_pressure_min and check_rot_temperature_min:
            self.debug(
                'Min pressure and rotational temperature reached. Interpolating on vibrational temperatures only')
            return self.interp_vib_temp_only(Tv, tv_idx_min, tv_idx_max, 0, 0, wngrid_filter).ravel()

        if check_pressure_min and check_vib_temperature_min:
            self.debug(
                'Min pressure and vibrational temperature reached. Interpolating on rotational temperatures only')
            return self.interp_rot_temp_only(Tr, tr_idx_min, tr_idx_max, 0, 0, wngrid_filter).ravel()

        if check_rot_temperature_min and check_vib_temperature_min:
            self.debug('Min temperatures reached on both accounts. Interpolating pressure only')
            return self.interp_pressure_only(P, p_idx_min, p_idx_max, 0, wngrid_filter, Tv=0).ravel()

        if check_pressure_max and check_rot_temperature_min:
            self.debug(
                'Max pressure and min rotational temperature reached. Interpolating on vibrational temperatures only')
            return self.interp_vib_temp_only(Tv, tv_idx_min, tv_idx_max, 0, -1, wngrid_filter).ravel()

        if check_pressure_max and check_vib_temperature_min:
            self.debug(
                'Max pressure and min vibrational temperature reached. Interpolating on rotational temperatures only')
            return self.interp_rot_temp_only(Tr, tr_idx_min, tr_idx_max, 0, -1, wngrid_filter).ravel()

        if check_rot_temperature_max and check_vib_temperature_min:
            self.debug(
                'Max rot temperature and min vib temperature reached on both accounts. Interpolating pressure only')
            return self.interp_pressure_only(P, p_idx_min, p_idx_max, -1, wngrid_filter, Tv=0).ravel()

        if check_pressure_min and check_rot_temperature_max:
            self.debug(
                'Min pressure and Max rotational temperature reached. Interpolating on vibrational temperatures only')
            return self.interp_vib_temp_only(Tv, tv_idx_min, tv_idx_max, -1, 0, wngrid_filter).ravel()

        if check_pressure_min and check_vib_temperature_max:
            self.debug(
                'Min pressure and max vibrational temperature reached. Interpolating on rotational temperatures only')
            return self.interp_rot_temp_only(Tr, tr_idx_min, tr_idx_max, -1, 0, wngrid_filter).ravel()

        if check_rot_temperature_min and check_vib_temperature_max:
            self.debug(
                'Min rot temperature and max vib temperature reached on both accounts. Interpolating pressure only')
            return self.interp_pressure_only(P, p_idx_min, p_idx_max, 0, wngrid_filter, Tv=-1).ravel()

        if check_pressure_min:
            self.debug('Min pressure reached. Interpolating on both temperatures only')
            return self.interp_two_temps(Tr, tr_idx_min, tr_idx_max, Tv, tv_idx_min, tv_idx_max, 0, wngrid_filter)

        # Max pressure
        if check_pressure_max:
            self.debug('Max pressure reached. Interpolating temperature only')
            return self.interp_two_temps(Tr, tr_idx_min, tr_idx_max, Tv, tv_idx_min, tv_idx_max, -1, wngrid_filter)

        if check_vib_temperature_min:
            self.debug("Minimum Vib Temperature Reached")
            return self.interp_pressure_rot_temp(0, Tr, tr_idx_min, tr_idx_max, P, p_idx_min, p_idx_max, wngrid_filter)

        if check_vib_temperature_max:
            self.debug("Maximum Vib Temperature Reached")
            return self.interp_pressure_rot_temp(-1, Tr, tr_idx_min, tr_idx_max, P, p_idx_min, p_idx_max, wngrid_filter)

        if check_rot_temperature_min:
            self.debug("Minimum Rot Temperature Reached")
            return self.interp_pressure_vib_temp(0, Tv, tv_idx_min, tv_idx_max, P, p_idx_min, p_idx_max, wngrid_filter)

        if check_rot_temperature_max:
            self.debug("Maximum Rot Temperature Reached")
            return self.interp_pressure_vib_temp(-1, Tv, tv_idx_min, tv_idx_max, P, p_idx_min, p_idx_max, wngrid_filter)

        q_111 = self.xsecGrid[p_idx_min, tr_idx_min, tv_idx_min][wngrid_filter].ravel()
        q_112 = self.xsecGrid[p_idx_min, tr_idx_min, tv_idx_max][wngrid_filter].ravel()
        q_121 = self.xsecGrid[p_idx_min, tr_idx_max, tv_idx_min][wngrid_filter].ravel()
        q_122 = self.xsecGrid[p_idx_min, tr_idx_max, tv_idx_max][wngrid_filter].ravel()
        q_211 = self.xsecGrid[p_idx_max, tr_idx_min, tv_idx_min][wngrid_filter].ravel()
        q_221 = self.xsecGrid[p_idx_max, tr_idx_max, tv_idx_min][wngrid_filter].ravel()
        q_222 = self.xsecGrid[p_idx_max, tr_idx_max, tv_idx_max][wngrid_filter].ravel()
        q_212 = self.xsecGrid[p_idx_max, tr_idx_min, tv_idx_max][wngrid_filter].ravel()

        Trmax = self.temperatureGrid[tr_idx_max]
        Trmin = self.temperatureGrid[tr_idx_min]
        Tvmax = self.vibTemperatureGrid[tv_idx_max]
        Tvmin = self.vibTemperatureGrid[tv_idx_min]
        Pmax = self.logPressure[p_idx_max]
        Pmin = self.logPressure[p_idx_min]

        if self._interp_mode == 'linear':
            return interp_trilin(q_111, q_112, q_121, q_122, q_211, q_212, q_221, q_222, Tr, Trmin, Trmax, Tv, Tvmin,
                                 Tvmax, P, Pmin, Pmax)
        elif self._interp_mode == 'exp':
            raise ValueError(
                'Unsupported interpolation mode {} for non-LTE'.format(self._interp_mode))
        else:
            raise ValueError(
                'Unknown interpolation mode {}'.format(self._interp_mode))

    def compute_opacity(self, temperature, pressure, wngrid=None, temperature_vib=None):
        import math
        logpressure = math.log10(pressure)
        if self.vibTemperatureGrid is not None:
            temp_vib = temperature_vib or temperature  # If no temp vib given then take as the same as rotational and
            # handle as LTE
            return self.interp_nonlte(temperature, temp_vib, logpressure,
                                      *self.find_closest_index(temperature, logpressure, Tv=temp_vib), wngrid) / 10000
        else:
            return self.interp_bilinear_grid(temperature, logpressure,
                                             *self.find_closest_index(temperature, logpressure), wngrid) / 10000

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
