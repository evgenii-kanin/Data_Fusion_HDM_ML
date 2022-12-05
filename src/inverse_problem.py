import numpy as np
import pandas as pd
import pickle
from scipy.optimize import curve_fit, differential_evolution
from sklearn.preprocessing import StandardScaler
from src.dataset_generation import cubes_around_wells, perm_calc_kr, approx_neighbour_perm


class InverseProblemSolution:
    def __init__(self, path_dataset, path_model_main, path_save_list):
        data_egg = pd.read_excel(path_dataset).iloc[:, :8]
        self.num_wells = data_egg.shape[0]
        self.num_cells = 15 * 15
        self.estimator = pickle.load(open(path_model_main, 'rb'))
        self.array_wells = np.array(data_egg[['X', 'Y']])
        self.array_coord, self.array_dist, self.array_coord_fit = self.mesh_func()
        self.bounds = self.set_bounds()
        self.target_values = np.array(data_egg[['k_well_log, mD', 'k_well_test, mD', 'skin']])
        self.path_save_perm, self.path_save_params, self.path_save_perm_wells = path_save_list

    def set_bounds(self):
        bounds_dict = {
            'rd': [100, 300],
            'rg': [5, 50],
            'alpha': [0.5, 2.],
            'beta': [1, 2.],
            'gamma': [0.01, 2],
            'delta': [0.05, 1],
            'k_far': [1, 15]
        }

        bounds_list = list(bounds_dict.values())

        for i in range(2 * self.num_wells - 1):
            bounds_list.append(bounds_dict['k_far'])

        return bounds_list

    def mesh_func(self):
        x_array_edge = np.linspace(-500, 2500, 91)
        x_array = 0.5 * (x_array_edge[1:] + x_array_edge[:-1])
        array_coord = cubes_around_wells(x_array, self.array_wells, self.num_wells, self.num_cells)
        array_dist = np.zeros((self.num_wells, self.num_cells, self.num_wells))
        array_coord_fit = np.zeros((self.num_wells, 2, self.num_cells))

        for i in np.arange(self.num_wells):
            array_dist[i] = np.linalg.norm(array_coord[i] - self.array_wells[:, None], axis=2).T
            array_coord_fit[i] = np.array([array_coord[i, :, 0] - self.array_wells[i, 0],
                                           array_coord[i, :, 1] - self.array_wells[i, 1]])

        return array_coord, array_dist, array_coord_fit

    def direct_problem(self, x, output=False):
        pred_values = np.zeros((self.num_wells, 3))

        kr_params, perm_params = x[:6], np.concatenate((x[6:18][:, None], x[18:30][:, None]), axis=1)

        for item in np.arange(self.num_wells):
            array_perm_total, array_perm_add, _ = \
                perm_calc_kr(item, self.array_dist[item], kr_params, perm_params, self.num_wells)

            pred_values[item, 0] = array_perm_total[:self.num_cells // 2 + 1][-1]

            result_fit = curve_fit(approx_neighbour_perm, self.array_coord_fit[item], array_perm_add)[0]

            pred_values[item, 1], pred_values[item, 2] = \
                self.estimator.predict(np.array([*kr_params,
                                                 perm_params[item, 0], perm_params[item, 1],
                                                 *result_fit])[None, :])[0]

        scaler_func = StandardScaler()
        scaler_func.fit(self.target_values)
        target_values_scaled = scaler_func.transform(self.target_values)
        pred_values_scaled = scaler_func.transform(pred_values[:, :3])

        result = np.sum(np.mean(abs(pred_values_scaled - target_values_scaled), axis=0))

        if output:
            return result, pred_values
        else:
            return result

    def minimization_func(self, num_iter):
        solution = differential_evolution(self.direct_problem, self.bounds, tol=1e-3, maxiter=num_iter).x
        _, pred_values = self.direct_problem(solution, True)
        np.save(self.path_save_perm_wells, pred_values)
        return solution, pred_values

    def write_perm_inc(self, sol):
        x_array_edge = np.linspace(0, 2000, 61)
        x_array = 0.5 * (x_array_edge[1:] + x_array_edge[:-1])
        xv, yv = np.meshgrid(x_array, x_array, indexing='xy')
        array_coord = np.array([xv.flatten(), yv.flatten()]).T

        dist = np.linalg.norm(array_coord - self.array_wells[:, None], axis=2).T

        kr_params, perm_params = sol[:6], np.concatenate((sol[6:18][:, None],
                                                          sol[18:30][:, None]), axis=1)

        array_perm_total = perm_calc_kr(0, dist, kr_params, perm_params, self.num_wells)[0]

        np.save(self.path_save_params, np.concatenate((kr_params, perm_params[:, 0], perm_params[:, 1])))

        with open(self.path_save_perm, 'w+') as file:
            file.write('PERMX' + '\n')
            file.write(' '.join(list(map(str, np.round(array_perm_total, 4)))) + '\n')
            file.write('/')

    def __call__(self, num_iter):
        solution, pred_values = self.minimization_func(num_iter)
        self.write_perm_inc(solution)
        return pred_values
