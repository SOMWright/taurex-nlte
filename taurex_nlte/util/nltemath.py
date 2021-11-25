import numba
import numpy as np


@numba.njit(nogil=True, fastmath=True)
def interp_trilin(c000, c001, c010, c011, c100, c101, c110, c111, Tr, Trmin, Trmax, Tv, Tvmin, Tvmax, P, Pmin, Pmax):
    # x -> P
    # y -> Trot
    # z-> Tvib

    factor = 1.0 / ((Pmin - Pmax) * (Trmin - Trmax) * (Tvmin - Tvmax))

    a0 = ((-1 * c000 * Pmax * Trmax * Tvmax) + (c001 * Pmax * Trmax * Tvmin) + (c010 * Pmax * Trmin * Tvmax) - (
            c011 * Pmax * Trmin * Tvmin)) + (
                 (c100 * Pmin * Trmax * Tvmax) - (c101 * Pmin * Trmax * Tvmin) - (c110 * Pmin * Trmin * Tvmax) + (
                 c111 * Pmin * Trmin * Tvmin))

    a1 = ((c000 * Trmax * Tvmax) - (c001 * Trmax * Tvmin) - (c010 * Trmin * Tvmax) + (c011 * Trmin * Tvmin)) + (
            -1 * (c100 * Trmax * Tvmax) + (c101 * Trmax * Tvmin) + (c110 * Trmin * Tvmax) - (c111 * Trmin * Tvmin))

    a2 = ((c000 * Pmax * Tvmax) - (c001 * Pmax * Tvmin) - (c010 * Pmax * Tvmax) + (c011 * Pmax * Tvmin)) + (
            (-1 * c100 * Pmin * Tvmax) + (c101 * Pmin * Tvmin) + (c110 * Pmin * Tvmax) - (c111 * Pmin * Tvmin))

    a3 = ((c000 * Pmax * Trmax) - (c001 * Pmax * Trmax) - (c010 * Pmax * Trmin) + (c011 * Pmax * Trmin)) + (
            (-1 * c100 * Pmin * Trmax) + (c101 * Pmin * Trmax) + (c110 * Pmin * Trmin) - (c111 * Pmin * Trmin))

    a4 = ((-1 * c000 * Tvmax) + (c001 * Tvmin) + (c010 * Tvmax) - (c011 * Tvmin) + (c100 * Tvmax) - (c101 * Tvmin) - (
            c110 * Tvmax) + (c111 * Tvmin))

    a5 = ((-1 * c000 * Trmax) + (c001 * Trmax) + (c010 * Trmin) - (c011 * Trmin) + (c100 * Trmax) - (c101 * Trmax) - (
            c110 * Trmin) + (c111 * Trmin))

    a6 = ((-1 * c000 * Pmax) + (c001 * Pmax) + (c010 * Pmax) - (c011 * Pmax) + (c100 * Pmin) - (c101 * Pmin) - (
            c110 * Pmin) + (c111 * Pmin))

    a7 = c000 - c001 - c010 + c011 - c100 + c101 + c110 - c111

    res = np.zeros_like(c000)
    for i in range(c000.shape[0]):
        res[i] = (a0[i] + (a1[i] * P) + (a2[i] * Tr) + (a3[i] * Tv) + (a4[i] * P * Tr) + (a5[i] * P * Tv) + (
                a6[i] * Tr * Tv) + (a7[i] * P * Tv * Tr)) * factor
    return res
