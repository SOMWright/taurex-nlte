import re
from taurex.util.util import tokenize_molecule, split_molecule_elements, merge_elements


def NLTE_mol_split_func(molecule=None, tokens=None):
    molecule = re.sub("NLTE", "", molecule, flags=re.IGNORECASE)
    from taurex.util.util import mass
    elems = {}

    if molecule:
        tokens = tokenize_molecule(molecule)

    length = 0

    while length < len(tokens):
        token = tokens[length]

        if token in mass:
            if token not in elems:
                elems[token] = 0
            try:
                peek = int(tokens[length + 1])

                length += 1
            except IndexError:
                peek = 1
            except ValueError:
                peek = 1
            elems[token] += peek
        elif token in '{([':
            length += 1
            sub_elems, moved = split_molecule_elements(tokens=tokens[length:])
            length += moved
            try:
                peek = int(tokens[length + 1])
                length += 1
            except IndexError:
                peek = 1
            except ValueError:
                peek = 1
            elems = merge_elements(elems, sub_elems, peek)
        elif token in '}])':
            return elems, length
        length += 1

    return elems

def generate_profile_dict_with_bitemp(model):
    out = {}
    out['temp_profile']=model.temperatureProfile
    if model._temperature_profile._log_name == "taurex.NLTETempProfile":
        out['nlte_temp_profile']=model._temperature_profile.nlte_profile
    out['active_mix_profile']=model.chemistry.activeGasMixProfile
    out['inactive_mix_profile']=model.chemistry.inactiveGasMixProfile
    out['density_profile']=model.densityProfile
    out['scaleheight_profile']=model.scaleheight_profile
    out['altitude_profile']=model.altitudeProfile
    out['gravity_profile']=model.gravity_profile
    out['pressure_profile']=model.pressureProfile
    if model.chemistry.hasCondensates:
        out['condensate_profile'] = model.chemistry.condensateMixProfile
    return out