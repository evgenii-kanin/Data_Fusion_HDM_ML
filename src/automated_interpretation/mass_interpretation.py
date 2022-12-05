import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from src.automated_interpretation.direct_solver import direct_problem_solution


def run_interpretation(bounds, target_names, params, pwf_ref, x0=None):
    res_opt = minimize(direct_problem_solution,
                       x0=x0,
                       args=(target_names, params, pwf_ref,),
                       bounds=bounds, tol=1e-3,
                       method='Nelder-Mead')

    return res_opt


def single_interpretation(bhp, kernel_params, time_array_mufits, time_array, target_names, params, x0=None):
    bounds = [(0.5, 30), (-6, 6)]

    if x0 is None:
        skin_0 = np.log(kernel_params[1] / 0.1) * (kernel_params[7] / kernel_params[6] - 1)

        if abs(skin_0) >= 6:
            skin_0 = 0

        x0 = [0.5 * (kernel_params[6] + kernel_params[7]), skin_0]

    bhp_interp = interp1d(time_array_mufits, bhp, kind='cubic')(time_array)

    calc = run_interpretation(bounds, target_names, params, bhp_interp, x0)

    if calc.success is True:
        result = list(calc.x)
    else:
        x0 = [7.5, 0]
        calc = run_interpretation(bounds, target_names, params, bhp_interp, x0)

        if calc.success is True:
            result = list(calc.x)
        else:
            result = [0, 0]

    return result


def mass_interpretation(file_name_mufits, file_name_params, path_save,
                        time_array_mufits, time_array, target_names, params):
    with open(file_name_mufits, 'r') as file:
        data = file.readlines()

    bhp_array = np.zeros((len(data) - 1, len(data[0].split()[1:77])))

    for num, line in enumerate(data[1:]):
        line_cur = line.split()
        bhp_array[int(line_cur[78]) - 1, :] = line_cur[1:77]

    kernel_params_array = np.load(file_name_params)

    filter_bhp = bhp_array[:, -1] > 10.1
    bhp_array, kernel_params_array = bhp_array[filter_bhp] / 1.01325, kernel_params_array[filter_bhp]

    results_list = Parallel(n_jobs=-1)(delayed(single_interpretation)(bhp_array[i], kernel_params_array[i],
                                                                      time_array_mufits, time_array, target_names,
                                                                      params) for i in range(bhp_array.shape[0]))

    kernel_params_array = np.concatenate((kernel_params_array, np.array(results_list)), axis=1)

    np.save(path_save, kernel_params_array)
