import numpy as np
from enum import Enum
from taurex.temperature import TemperatureProfile
from taurex.data.fittable import fitparam
from taurex.util import movingaverage
from taurex.exceptions import InvalidModelException


class TemperatureTypeEnum(Enum):
    ROTATIONAL = 1
    VIBRATIONAL = 2


class InvalidTemperatureException(InvalidModelException):
    """
    Exception that is called when atmosphere mix is greater
    than unity
    """
    pass


class NLTETempProfile(TemperatureProfile):

    def __init__(self, Trot_surface=1500.0, Trot_top=200.0, Tvib_surface=1500.0, Tvib_top=200.0, P_surface=None,
                 P_top=None, trot_points=[], tvib_points=[], pressure_points=[],
                 smoothing_window=10, limit_slope=9999999, physics_temperature="Rotational"):
        super().__init__('NLTETempProfile')

        self.info('Non-LTE Bi-temperature profile is initialized')
        self.debug('Passed vibrational temperature points %s', tvib_points)
        self.debug('Passed rotational temperature points %s', trot_points)
        self.debug('Passed pressure points %s', pressure_points)
        self._assign_profile_temp_types(physics_temperature)
        self._vib_t_points = tvib_points
        self._rot_t_points = trot_points
        self._p_points = pressure_points
        self._vib_T_surface = Tvib_surface
        self._rot_T_surface = Trot_surface
        self._vib_T_top = Tvib_top
        self._rot_T_top = Trot_top
        self._P_surface = P_surface
        self._P_top = P_top
        self._smooth_window = smoothing_window
        self._limit_slope = limit_slope
        self.generate_pressure_fitting_params()
        self.generate_vib_temperature_fitting_params()
        self.generate_rot_temperature_fitting_params()

    def _assign_profile_temp_types(self, physics_temperature: str) -> None:
        if physics_temperature.lower() == 'vibrational' or physics_temperature.lower() == 'vib':
            self.info("Taking vibrational temperature for physics temperature")
            self._profile_temp_type = TemperatureTypeEnum.VIBRATIONAL
            self._nlte_profile_temp_type = TemperatureTypeEnum.ROTATIONAL
        else:
            self.info("Taking rotational temperature for physics temperature")
            self._profile_temp_type = TemperatureTypeEnum.ROTATIONAL
            self._nlte_profile_temp_type = TemperatureTypeEnum.VIBRATIONAL

    @fitparam(param_name='Tvib_surface',
              param_latex='$Tvib_\\mathrm{surf}$',
              default_fit=False,
              default_bounds=[300, 2500])
    def temperatureVibSurface(self):
        """Vibrational Temperature at planet surface in Kelvin"""
        return self._vib_T_surface

    @temperatureVibSurface.setter
    def temperatureVibSurface(self, value):
        self._vib_T_surface = value

    @fitparam(param_name='Trot_surface',
              param_latex='$Trot_\\mathrm{surf}$',
              default_fit=False,
              default_bounds=[300, 2500])
    def temperatureRotSurface(self):
        """Rotational Temperature at planet surface in Kelvin"""
        return self._rot_T_surface

    @temperatureRotSurface.setter
    def temperatureRotSurface(self, value):
        self._rot_T_surface = value

    @fitparam(param_name='Tvib_top',
              param_latex='$Tvib_\\mathrm{top}$',
              default_fit=False,
              default_bounds=[300, 2500])
    def temperatureVibTop(self):
        """Vibrational Temperature at top of atmosphere in Kelvin"""
        return self._vib_T_top

    @temperatureVibTop.setter
    def temperatureVibTop(self, value):
        self._vib_T_top = value

    @fitparam(param_name='Trot_top',
              param_latex='$Trot_\\mathrm{top}$',
              default_fit=False,
              default_bounds=[300, 2500])
    def temperatureRotTop(self):
        """Rotational Temperature at top of atmosphere in Kelvin"""
        return self._rot_T_top

    @temperatureRotTop.setter
    def temperatureRotTop(self, value):
        self._rot_T_top = value

    @fitparam(param_name='P_surface',
              param_latex='$P_\\mathrm{surf}$',
              default_fit=False,
              default_bounds=[1e3, 1e2],
              default_mode='log')
    def pressureSurface(self):
        return self._P_surface

    @pressureSurface.setter
    def pressureSurface(self, value):
        self._P_surface = value

    @fitparam(param_name='P_top',
              param_latex='$P_\\mathrm{top}$',
              default_fit=False,
              default_bounds=[1e-5, 1e-4],
              default_mode='log')
    def pressureTop(self):
        return self._P_top

    @pressureTop.setter
    def pressureTop(self, value):
        self._P_top = value

    def generate_pressure_fitting_params(self):
        """Generates the fitting parameters for the pressure points
        These are given the name ``P_point(number)`` for example, if two extra
        pressure points are defined between the top and surface then the
        fitting parameters generated are ``P_point0`` and ``P_point1``
        """

        bounds = [1e5, 1e3]
        for idx, val in enumerate(self._p_points):
            point_num = idx + 1
            param_name = 'P_point{}'.format(point_num)
            param_latex = '$P_{}$'.format(point_num)

            def read_point(self, idx=idx):
                return self._p_points[idx]

            def write_point(self, value, idx=idx):
                self._p_points[idx] = value

            fget_point = read_point
            fset_point = write_point
            self.debug('FGet_location %s', fget_point)
            default_fit = False
            self.add_fittable_param(param_name, param_latex, fget_point,
                                    fset_point, 'log', default_fit, bounds)

    def generate_vib_temperature_fitting_params(self):
        """Generates the fitting parameters for the vibrational temperature points
        These are given the name ``Vib_T_point(number)`` for example, if two extra
        temperature points are defined between the top and surface then the
        fitting parameters generated are ``Vib_T_point0`` and ``Vib_T_point1``
        """

        bounds = [300, 2500]
        for idx, val in enumerate(self._vib_t_points):
            point_num = idx + 1
            param_name = 'Vib_T_point{}'.format(point_num)
            param_latex = '$Vib_T_{}$'.format(point_num)

            def read_point(self, idx=idx):
                return self._vib_t_points[idx]

            def write_point(self, value, idx=idx):
                self._vib_t_points[idx] = value

            fget_point = read_point
            fset_point = write_point
            self.debug('FGet_location %s %s', fget_point, fget_point(self))
            default_fit = False
            self.add_fittable_param(param_name, param_latex, fget_point,
                                    fset_point, 'linear', default_fit, bounds)

    def generate_rot_temperature_fitting_params(self):
        """Generates the fitting parameters for the rotational temperature points
        These are given the name ``rot_T_point(number)`` for example, if two extra
        temperature points are defined between the top and surface then the
        fitting parameters generated are ``Rot_T_point0`` and ``Rot_T_point1``
        """

        bounds = [300, 2500]
        for idx, val in enumerate(self._rot_t_points):
            point_num = idx + 1
            param_name = 'Rot_T_point{}'.format(point_num)
            param_latex = '$Rot_T_{}$'.format(point_num)

            def read_point(self, idx=idx):
                return self._vib_t_points[idx]

            def write_point(self, value, idx=idx):
                self._rot_t_points[idx] = value

            fget_point = read_point
            fset_point = write_point
            self.debug('FGet_location %s %s', fget_point, fget_point(self))
            default_fit = False
            self.add_fittable_param(param_name, param_latex, fget_point,
                                    fset_point, 'linear', default_fit, bounds)

    def check_profile(self, Ppt, Tpt):

        if (any(Ppt[i] <= Ppt[i + 1] for i in range(len(Ppt) - 1))):
            self.warning('Temperature profile is not valid - a pressure point is inverted')
            raise InvalidTemperatureException

        if (any(abs((Tpt[i + 1] - Tpt[i]) / (np.log10(Ppt[i + 1]) - np.log10(Ppt[i]))) >= self._limit_slope for i in
                range(len(Ppt) - 1))):
            self.warning('Temperature profile is not valid - profile slope too high')
            raise InvalidTemperatureException

    def profile_calculation(self, T_surface, T_points, T_top, P_surface, P_points, P_top):

        Tnodes = [T_surface, *T_points, T_top]

        if P_surface is None or P_surface < 0:
            P_surface = self.pressure_profile[0]

        if P_top is None or P_top < 0:
            P_top = self.pressure_profile[-1]

        Pnodes = [P_surface, *P_points, P_top]

        self.check_profile(Pnodes, Tnodes)

        if np.all(Tnodes == Tnodes[0]):
            return np.ones_like(self.pressure_profile) * Tnodes[0]


        profile = np.interp((np.log10(self.pressure_profile[::-1])),
                           np.log10(Pnodes[::-1]), Tnodes)

        return profile

    @property
    def nlte_profile(self):
        if self._nlte_profile_temp_type is TemperatureTypeEnum.ROTATIONAL:
            return self.profile_calculation(self._rot_T_surface, self._rot_t_points, self._rot_T_top, self._P_surface,
                                            self._p_points, self._P_top)
        else:
            return self.profile_calculation(self._vib_T_surface, self._vib_t_points, self._vib_T_top, self._P_surface,
                                            self._p_points, self._P_top)

    @property
    def profile_temp_type(self):
        return self._profile_temp_type

    @property
    def nlte_profile_temp_type(self):
        return self._nlte_profile_temp_type

    @property
    def profile(self):

        if self._profile_temp_type is TemperatureTypeEnum.ROTATIONAL:
            return self.profile_calculation(self._rot_T_surface, self._rot_t_points, self._rot_T_top, self._P_surface,
                                            self._p_points, self._P_top)
        else:
            return self.profile_calculation(self._vib_T_surface, self._vib_t_points, self._vib_T_top, self._P_surface,
                                            self._p_points, self._P_top)

    def write(self, output):
        temperature = super().write(output)

        temperature.write_scalar('Vib_T_surface', self._vib_T_surface)
        temperature.write_scalar('Vib_T_top', self._vib_T_top)
        temperature.write_scalar('Rot_T_surface', self._rot_T_surface)
        temperature.write_scalar('Rot_T_top', self._rot_T_top)
        temperature.write_array('rot_temperature_points', np.array(self._rot_t_points))
        temperature.write_array('vib_temperature_points', np.array(self._vib_t_points))

        P_surface = self._P_surface
        P_top = self._P_top
        if not P_surface:
            P_surface = -1
        if not P_top:
            P_top = -1

        temperature.write_scalar('P_surface', P_surface)
        temperature.write_scalar('P_top', P_top)
        temperature.write_array('pressure_points', np.array(self._p_points))

        temperature.write_scalar('physics_temperature', self.profile_temp_type.name)
        temperature.write_scalar('nlte_temperature', self.nlte_profile_temp_type.name)

        return temperature

    @classmethod
    def input_keywords(cls):
        """
        Return all input keywords
        """
        return ['nltetempprofile', 'non-lte', 'bitemp', '2TP']
