import numpy as np
from taurex.contributions import AbsorptionContribution
from taurex.cache.opacitycache import OpacityCache

#Could potentially use a Mixin here, potentially
class NLTEAbsorption(AbsorptionContribution):
    def __init__(self, rot_offset=None, vib_offset=None):
        super().__init__('Absorption')
        self._opacity_cache = OpacityCache()
        self._rot_offset = rot_offset
        self._vib_offset = vib_offset

        if self._rot_offset is not None:
            self.add_rot_fitting_param()

        if self._vib_offset is not None:
            self.add_vib_fitting_param()

    def add_rot_fitting_param(self):
        self.add_fittable_param("rot_offset", "$T_{rot}$ offset", self.rot_offset, self.set_rot_offset, 'linear', False,
                                [-500, 0])

    def rot_offset(self):
        return self._rot_offset

    def set_rot_offset(self, rot_offset):
        self._rot_offset = rot_offset

    def add_vib_fitting_param(self):
        self.add_fittable_param("vib_offset", "$T_{vib}$ offset", self.vib_offset, self.set_vib_offset, 'linear', False,
                                [0, 500])

    def vib_offset(self):
        return self._vib_offset

    def set_vib_offset(self, vib_offset):
        self._vib_offset = vib_offset
