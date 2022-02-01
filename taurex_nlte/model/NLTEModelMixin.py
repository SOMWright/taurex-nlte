import numpy as np
from taurex.mixin import ForwardModelMixin


class NLTEModelMixin(ForwardModelMixin):

    def __init_mixin__(self):
        pass

    def compute_error(self, samples, wngrid=None, binner=None):
        """

        Computes standard deviations from samples

        Parameters
        ----------

        samples:

        """
        from taurex.util.math import OnlineVariance
        tp_profiles = OnlineVariance()
        nlte_tp_profiles = OnlineVariance()
        active_gases = OnlineVariance()
        inactive_gases = OnlineVariance()
        cond = None

        has_condensates = self.chemistry.hasCondensates

        if has_condensates:
            cond = OnlineVariance()

        if binner is not None:
            binned_spectrum = OnlineVariance()
        else:
            binned_spectrum = None
        native_spectrum = OnlineVariance()

        for weight in samples():

            native_grid, native, tau, _ = self.model(wngrid=wngrid,
                                                     cutoff_grid=False)

            tp_profiles.update(self.temperatureProfile, weight=weight)
            if self._temperature_profile._log_name == "taurex.NLTETempProfile":
                nlte_tp_profiles.update(self._temperature_profile.nlte_profile, weight=weight)
            active_gases.update(self.chemistry.activeGasMixProfile,
                                weight=weight)
            inactive_gases.update(self.chemistry.inactiveGasMixProfile,
                                  weight=weight)

            if cond is not None:
                cond.update(self.chemistry.condensateMixProfile,
                            weight=weight)

            native_spectrum.update(native, weight=weight)

            if binned_spectrum is not None:
                binned = binner.bindown(native_grid, native)[1]
                binned_spectrum.update(binned, weight=weight)

        profile_dict = {}
        spectrum_dict = {}

        tp_std = np.sqrt(tp_profiles.parallelVariance())
        nlte_tp_std = np.sqrt(nlte_tp_profiles.parallelVariance())
        active_std = np.sqrt(active_gases.parallelVariance())
        inactive_std = np.sqrt(inactive_gases.parallelVariance())

        profile_dict['temp_profile_std'] = tp_std
        profile_dict['nlte_temp_profile_std'] = nlte_tp_std
        profile_dict['active_mix_profile_std'] = active_std
        profile_dict['inactive_mix_profile_std'] = inactive_std
        if cond is not None:
            profile_dict['condensate_profile_std'] = \
                np.sqrt(cond.parallelVariance())

        spectrum_dict['native_std'] = \
            np.sqrt(native_spectrum.parallelVariance())

        if binned_spectrum is not None:
            spectrum_dict['binned_std'] = \
                np.sqrt(binned_spectrum.parallelVariance())

        return profile_dict, spectrum_dict

    @classmethod
    def input_keywords(self):
        return ['nlte', 'non-LTE', 'bitemp']
