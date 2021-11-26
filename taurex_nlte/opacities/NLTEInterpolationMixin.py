import numpy as np
from taurex.util.math import intepr_bilin, interp_exp_and_lin, interp_lin_only, interp_exp_only
from ..util.nltemath import interp_trilin


class NLTEInterpolationMixin():

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
