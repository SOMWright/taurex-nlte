from taurex.data.profiles.chemistry.gas.gas import Gas
from taurex.util import movingaverage, molecule_texlabel
import numpy as np


class NPointGas(Gas):

    def __init__(self, molecule_name='OH', mix_ratio_surface=1e-4,
                 mix_ratio_top=1e-8, mix_ratios=[], p_points=[]):
        super().__init__(self.__class__.__name__, molecule_name=molecule_name)

        self._mix_surface = mix_ratio_surface
        self._mix_top = mix_ratio_top
        self._mix_ratios = mix_ratios
        self._p_points = p_points
        self._mix_profile = None
        self.add_surface_param()
        self.add_top_param()
        self.generate_pressure_fitting_params()
        self.generate_mix_ratio_fitting_params()

    @property
    def mixProfile(self):
        """

        Returns
        -------
        mix: :obj:`array`
            Mix ratio for molecule at each layer

        """
        return self._mix_profile

    @property
    def mixRatioSurface(self):
        """Abundance on the planets surface"""
        return self._mix_surface

    @property
    def mixRatioTop(self):
        """Abundance on the top of atmosphere"""
        return self._mix_top

    @property
    def mixRatios(self):
        return self._mix_ratios

    @property
    def pPoints(self):
        return self._p_points

    @mixRatioSurface.setter
    def mixRatioSurface(self, value):
        self._mix_surface = value

    @mixRatioTop.setter
    def mixRatioTop(self, value):
        self._mix_top = value

    def add_surface_param(self):
        """
        Generates surface fitting parameters. Has the form
        ''Moleculename_surface'
        """
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)

        param_surface = '{}_surface'.format(param_name)
        param_surf_tex = '{}_surface'.format(param_tex)

        def read_surf(self):
            return self._mix_surface

        def write_surf(self, value):
            self._mix_surface = value

        fget_surf = read_surf
        fset_surf = write_surf

        bounds = [1.0e-12, 0.1]

        default_fit = False
        self.add_fittable_param(param_surface, param_surf_tex, fget_surf,
                                fset_surf, 'log', default_fit, bounds)

    def add_top_param(self):
        """
        Generates TOA fitting parameters. Has the form:
        'Moleculename_top'
        """
        param_name = self.molecule
        param_tex = molecule_texlabel(param_name)

        param_top = '{}_top'.format(param_name)
        param_top_tex = '{}_top'.format(param_tex)

        def read_top(self):
            return self._mix_top

        def write_top(self, value):
            self._mix_top = value

        fget_top = read_top
        fset_top = write_top

        bounds = [1.0e-12, 0.1]

        default_fit = False
        self.add_fittable_param(param_top, param_top_tex, fget_top,
                                fset_top, 'log', default_fit, bounds)


    def generate_pressure_fitting_params(self):

        bounds = [1.0e-12, 0.1]
        for idx, val in enumerate(self._p_points):
            point_num = idx+1
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

    def generate_mix_ratio_fitting_params(self):

        bounds = [0.0, 1.0]
        for idx, val in enumerate(self._mix_ratios):
            point_num = idx+1
            param_name = 'Mix_ratio_point{}'.format(point_num)
            param_latex = '$Mix_ratio_{}$'.format(point_num)

            def read_point(self, idx=idx):
                return self._mix_ratios[idx]

            def write_point(self, value, idx=idx):
                self._mix_ratios[idx] = value

            fget_point = read_point
            fset_point = write_point
            self.debug('FGet_location %s %s', fget_point, fget_point(self))
            default_fit = False
            self.add_fittable_param(param_name, param_latex, fget_point,
                                    fset_point, 'linear', default_fit, bounds)


    def initialize_profile(self, nlayers=None, temperature_profile=None,
                           pressure_profile=None, altitude_profile=None):
        self._mix_profile = np.zeros(nlayers)

        pressure_layers = [pressure_profile[np.abs(pressure_profile - i).argmin()] for i in self._p_points]

        Pnodes = [pressure_profile[0], *pressure_layers, pressure_profile[-1]]

        Cnodes = [self.mixRatioSurface, *self._mix_ratios, self.mixRatioTop]

        chemprofile = 10**np.interp((np.log(pressure_profile[::-1])),
                                    np.log(Pnodes[::-1]),
                                    np.log10(Cnodes[::-1]))

        self._mix_profile = chemprofile[::-1]



    def write(self, output):
        gas_entry = super().write(output)
        gas_entry.write_scalar('mix_ratio_top', self.mixRatioTop)
        gas_entry.write_scalar('mix_ratio_surface', self.mixRatioSurface)
        gas_entry.write_array('mix_ratios', np.array(self.mixRatios))
        gas_entry.write_array('p_points', np.array(self.pPoints))

        return gas_entry

    @classmethod
    def input_keywords(self):
        return ['npoint', 'npointgas' ]