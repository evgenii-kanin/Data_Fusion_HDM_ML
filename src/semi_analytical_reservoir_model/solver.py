import numpy as np
from math import factorial
from src.semi_analytical_reservoir_model.model import pd_lapl_vert

NUMBER_OF_LAPL_COEFF = 10
DELTA_TIME_MIN = 1E-3
EPSILON = 1E-3


def run_func(params):
    result = np.zeros_like(params.td_array)
    for i in np.arange(params.td_array.shape[0]):
        result[i] = params.p_init - params.mult_p * pd_qd_calc(params.td_array[i], params)

    return result


def pd_qd_calc(td, params):
    weights = stehfest_coef(NUMBER_OF_LAPL_COEFF)

    result = 0

    for j in range(1, NUMBER_OF_LAPL_COEFF + 1):
        s = j * np.log(2) / td
        q_d = (2 * np.pi) / s * params.flow_rate
        p_d = pd_lapl_vert(s, params)
        add = (weights[j - 1] / j) * p_d * q_d * s
        result += add

    return result


def stehfest_coef(n):
    v_array = np.zeros(n)
    for i in range(1, n + 1):
        result = 0
        for k in range(int((i + 1) / 2), int(min(i, n / 2)) + 1):
            result += ((-1) ** (n / 2 + i) * k ** (n / 2) * factorial(2 * k) / factorial(n / 2 - k) /
                       factorial(k) / factorial(k - 1) / factorial(i - k) / factorial(2 * k - i))
        v_array[i - 1] = result

    return v_array
