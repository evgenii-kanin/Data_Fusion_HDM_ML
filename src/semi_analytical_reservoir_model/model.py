import numpy as np
from scipy.special import k0, k1

TINY = 1e-10
SMALL_BESSEL_ARG = 1e-4
MAX_NUMBER = 1000
PART_SUM_NUM = 5


def pd_lapl_vert(s, params):
    if np.allclose(params.yd, params.ywd):
        result = pd_b1(s, params) + pd_b2(s, params, True) + pd_b3(s, params)
    else:
        result = pd_b1(s, params, False) + pd_b2(s, params, False)

    result += params.skin / (2 * np.pi)

    return result


def pd_b1(s, params, subtract_inf=True):
    if params.xbound == "c":
        return 0

    signum = 1 if params.ybound == "n" else -1
    sbtr_inf = 0 if subtract_inf else 1

    u = s ** 0.5

    sum_exp = sumexp(u * params.yed)

    yd1 = params.yed - abs(params.yd - params.ywd)
    yd2 = params.yed - (params.yd + params.ywd)

    result = 1 / (2 * u * params.xed) * \
        (np.exp(-u * abs(params.yd - params.ywd)) * (sbtr_inf + sum_exp) +
         (signum * np.exp(-u * (params.yd + params.ywd)) + signum * np.exp(-u * (params.yed + yd2)) +
          np.exp(-u * (params.yed + yd1))) * (1 + sum_exp))

    return result


def pd_b2(s, params, subtract_inf=True):
    result = 0
    k = 1
    psumabs = PART_SUM_NUM

    while abs(psumabs) / PART_SUM_NUM >= TINY * (abs(result) + TINY) and k < MAX_NUMBER:
        psum = 0
        psumabs = 0
        for i in range(1, PART_SUM_NUM + 1):
            add = pd_b2_k(k, s, params, subtract_inf)
            psum += add
            psumabs += abs(add)
            k += 1
        result += psum

    return result


def pd_b2_k(k, s, params, subtract_inf):
    if params.xbound == "n":
        part_1 = 2 / params.xed * np.cos(k * np.pi * params.xd / params.xed) * \
                 np.cos(k * np.pi * params.xwd / params.xed)
    else:
        part_1 = 2 / params.xed * np.sin(k * np.pi * params.xd / params.xed) * \
                 np.sin(k * np.pi * params.xwd / params.xed)

    signum = 1 if params.ybound == "n" else -1
    sbtr_inf = 0 if subtract_inf else 1

    ek = np.sqrt(s + (k * np.pi / params.xed) ** 2)

    sum_exp = sumexp(ek * params.yed)

    yd1 = params.yed - abs(params.yd - params.ywd)
    yd2 = params.yed - (params.yd + params.ywd)

    part_2 = np.exp(-ek * abs(params.yd - params.ywd)) * (sbtr_inf + sum_exp) + \
        (signum * np.exp(-ek * (params.yd + params.ywd)) + np.exp(-ek * (params.yed + yd1)) +
            signum * np.exp(-ek * (params.yed + yd2))) * (1 + sum_exp)

    part_2 *= 1 / (2 * ek)

    return part_1 * part_2


def pd_b3(s, params):
    signum = 1 if params.xbound == "n" else -1

    result = pdb3_sub(0, s, params, -1, 1) + signum * pdb3_sub(0, s, params, 1, 1)

    k = 1
    add = 2 * result

    while abs(add) >= TINY * (abs(result) + TINY):
        add = signum * pdb3_sub(k, s, params, 1, 1) + pdb3_sub(k, s, params, -1, 1) + \
              signum * pdb3_sub(k, s, params, 1, -1, ) + pdb3_sub(k, s, params, -1, -1)
        result += add
        k += 1

    return result


def pdb3_sub(k, s, params, sgn1, sgn2):
    xd1 = abs(params.xd + sgn1 * params.xwd + sgn2 * 2 * k * params.xed)
    yd1 = abs(params.yd - params.ywd)

    rd = (xd1 ** 2 + yd1 ** 2) ** 0.5
    result = unit_cylinder_source(s, rd)

    return result


def unit_cylinder_source(s, r_d):
    u = s ** 0.5

    if u * r_d > 700:
        return 0

    if r_d * u / 2 > SMALL_BESSEL_ARG:
        if u < SMALL_BESSEL_ARG:
            result = 1 / (2 * np.pi) * k0(u * r_d)
        else:
            result = 1 / (2 * np.pi) * k0(u * r_d) / (u * k1(u))
    else:
        result = 1 / (2 * np.pi) * (-np.log(r_d * u / 2) - np.euler_gamma)

    return result


def sumexp(arg):
    result = 0
    m = 0
    inc = 1
    while inc > TINY * result:
        m += 1
        inc = np.exp(-2 * m * arg)
        result += inc
    return result
