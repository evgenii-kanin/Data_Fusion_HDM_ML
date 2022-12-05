import numpy as np
from src.semi_analytical_reservoir_model.solver import run_func
from src.semi_analytical_reservoir_model.properties import PropertiesClass


def direct_problem_solution(x, target_names, par, pwf_ref, output_pressure=False):
    par.update(dict(zip(target_names, x)))

    params = PropertiesClass(par['time_array'], par['flow_rate'],
                             par['mu'], par['b'], par['ct'],
                             par['k'], par['phi'],
                             par['p_init'], par['rw'],
                             par['portion_x'], par['portion_y'],
                             par['xe'], par['ye'], par['ze'],
                             par['xbound'], par['ybound'],
                             par['skin'])

    pres_calc = run_func(params)
    cost_func = np.sum((pwf_ref - pres_calc) ** 2)

    if output_pressure:
        return pres_calc, cost_func
    else:
        return cost_func
