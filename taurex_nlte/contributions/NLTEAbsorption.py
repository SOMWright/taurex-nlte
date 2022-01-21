import numpy as np
import taurex.util.util
from taurex.contributions import AbsorptionContribution
from taurex.cache import OpacityCache, GlobalCache
from ..util.util import NLTE_mol_split_func


class NLTEAbsorption(AbsorptionContribution):

    def __init__(self, rot_offset=None, vib_offset=None):
        taurex.util.util.split_molecule_elements = NLTE_mol_split_func
        super().__init__()
        self.info(
            "NLTE Absorption Selected: Overriding Molecule Split Utility Function to Support Non-LTE Molecule Naming")
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

    def prepare_each(self, model, wngrid):
        self.debug('Preparing model with %s', wngrid.shape)
        self._ngrid = wngrid.shape[0]
        self._use_ktables = GlobalCache()['opacity_method'] == 'ktables'
        self.info('Using Non-LTE cross-sections? %s', not self._use_ktables)
        weights = None

        if self._use_ktables:
            self.error("KTables are Not Support for Non-LTE")
        else:
            self._opacity_cache = OpacityCache()
        sigma_xsec = None
        self.weights = None

        for gas in model.chemistry.activeGases:

            # self._total_contrib[...] =0.0
            gas_mix = model.chemistry.get_gas_mix_profile(gas)
            self.info('Recomputing active gas %s opacity', gas)

            xsec = self._opacity_cache[gas]

            if self._use_ktables and self.weights is None:
                self.weights = xsec.weights

            if sigma_xsec is None:

                if self._use_ktables:
                    sigma_xsec = np.zeros(shape=(self._nlayers, self._ngrid, len(self.weights)))
                else:
                    sigma_xsec = np.zeros(shape=(self._nlayers, self._ngrid))
            else:
                sigma_xsec[...] = 0.0

            for idx_layer, tp in enumerate(zip(model.temperatureProfile, model.pressureProfile)):
                self.debug('Got index,tp %s %s', idx_layer, tp)
                temperature, pressure = tp
                # print(gas,self._opacity_cache[gas].opacity(temperature,pressure,wngrid),gas_mix[idx_layer])
                if self._rot_offset is not None:
                    sigma_xsec[idx_layer] += xsec.opacity(temperature + self._rot_offset, pressure, wngrid,
                                                          temperature_vib=temperature) * gas_mix[idx_layer]
                elif self._vib_offset is not None:
                    sigma_xsec[idx_layer] += xsec.opacity(temperature, pressure, wngrid,
                                                          temperature_vib=temperature + self._vib_offset) * gas_mix[
                                                 idx_layer]
                # elif model._temperature_profile._log_name == "taurex.NLTETempProfile":
                #     sigma_xsec[idx_layer] += xsec.opacity(temperature[0], pressure, wngrid,
                #                                           temperature_vib=temperature[1]) * gas_mix[idx_layer]
                else:
                    sigma_xsec[idx_layer] += xsec.opacity(temperature, pressure, wngrid) * gas_mix[idx_layer]

            self.sigma_xsec = sigma_xsec

            self.debug('SIGMAXSEC %s', self.sigma_xsec)

            yield gas, sigma_xsec

    @classmethod
    def input_keywords(self):
        return ['Non-LTE Absorption', 'NLTEAbsorption', ]
