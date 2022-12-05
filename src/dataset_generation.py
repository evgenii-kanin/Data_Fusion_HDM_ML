import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import glob
from scipy.interpolate import griddata


def calc_dist(new_well, list_wells, min_dist):
    array_dist = np.linalg.norm(new_well - np.array(list_wells), axis=1)
    return np.min(array_dist) >= min_dist


def denominator_numerator_func(r, kr_params, perm_params, opt):
    rd, rg, alpha, beta, gamma, delta = kr_params
    k_near, k_far = perm_params[:, 0], perm_params[:, 1]

    if opt == 1:
        return gamma * np.exp(-(r / rg) ** delta) + (r / rd) ** alpha * np.exp(-(r / rd) ** beta)
    elif opt == 2:
        return k_near * gamma * np.exp(-(r / rg) ** delta) + k_far * (r / rd) ** alpha * np.exp(-(r / rd) ** beta)


def perm_calc_kr(item, dist, kr_params, perm_params, num_wells):
    numerator = denominator_numerator_func(dist, kr_params, perm_params, 2)
    denominator = denominator_numerator_func(dist, kr_params, perm_params, 1)

    perm_total = np.sum(numerator, axis=1) / np.sum(denominator, axis=1)

    perm_add = np.sum(numerator[:, np.arange(num_wells) != item], axis=1) / \
        np.sum(denominator[:, np.arange(num_wells) != item], axis=1)

    perm_well = np.sum(numerator[:, np.arange(num_wells) == item], axis=1) / \
        np.sum(denominator[:, np.arange(num_wells) == item], axis=1)

    return perm_total, perm_add, perm_well


def approx_neighbour_perm(position, a, b, c, d, e, f):
    x, y = position
    return a * x ** 2 + b * x * y + c * y ** 2 + d * x + e * y + f


def generate_wells(coord_array, min_dist, max_iter=20, offset=250):
    list_wells = [[0, 0]]

    flag = True
    num_iter = 0

    while flag:
        while num_iter <= max_iter:
            new_well_index = np.random.randint(low=0, high=coord_array.shape[0], size=2)
            new_well = np.array([coord_array[new_well_index[0]], coord_array[new_well_index[1]]])

            num_iter += 1

            if np.all(np.abs(new_well) < max(coord_array) - offset) and calc_dist(new_well, list_wells, min_dist):
                list_wells.append(list(new_well))
                num_iter = 0
                break
        else:
            flag = False

    return np.array(list_wells)


def plot_wells(array_wells):
    plt.figure(figsize=(6, 6))

    coord = np.array([[-250, -250], [-250, 250], [250, 250], [250, -250], [-250, -250]])
    xs, ys = zip(*coord)

    for i in range(array_wells.shape[0]):
        plt.plot(array_wells[i, 0], array_wells[i, 1], marker='x', color='black', markersize=6)
        plt.plot(array_wells[i, 0] + xs, array_wells[i, 1] + ys, color='blue', linestyle=':', linewidth=2)

    plt.xlim(-2000, 2000)
    plt.ylim(-2000, 2000)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.axis('scaled')

    plt.xlabel('x, m', fontsize=15)
    plt.ylabel('y, m', fontsize=15)

    plt.grid()
    plt.tight_layout()

    plt.show()


def cubes_around_wells(x_array, array_wells, num_wells, num_cells):
    array_coord = np.zeros((num_wells, num_cells, 2))

    for item in np.arange(num_wells):
        xv, yv = np.meshgrid(x_array[np.abs(x_array - array_wells[item, 0]) <= 250],
                             x_array[np.abs(x_array - array_wells[item, 1]) <= 250], indexing='ij')
        array_coord[item] = np.array([xv.flatten(), yv.flatten()]).T

    return array_coord


def generate_perm_from_map(array_wells, x_array, num_cells, last_num=None, max_num=None, plot_flag=False):
    last_num = 1 if last_num is None else last_num
    max_num = 10_001 if max_num is None else max_num

    num_wells = array_wells.shape[0]

    alpha = np.random.uniform(low=0.5, high=2.01, size=num_wells)
    beta = np.random.uniform(low=1, high=2.01, size=num_wells)
    gamma = np.random.uniform(low=0.01, high=2.01, size=num_wells)
    delta = np.random.uniform(low=0.05, high=1.01, size=num_wells)

    rg = np.random.uniform(low=5, high=50.01, size=num_wells)
    rd = np.random.uniform(low=100, high=300.01, size=num_wells)

    array_coord = cubes_around_wells(x_array, array_wells, num_wells, num_cells)

    array_perm_total = np.zeros((num_wells, num_cells))
    array_perm_well = np.zeros((num_wells, num_cells))
    array_perm_add = np.zeros((num_wells, num_cells))
    array_kr_params = np.zeros((num_wells, 14))

    kr_params = np.array([rd, rg, alpha, beta, gamma, delta])
    perm_params = np.random.uniform(low=1, high=15.01, size=(num_wells, 2))

    array_kr_params[:, :6], array_kr_params[:, 6:8] = kr_params.T, perm_params

    for item in np.arange(num_wells):
        dist = np.linalg.norm(array_coord[item] - array_wells[:, None], axis=2).T

        array_perm_total[item], array_perm_add[item], array_perm_well[item] = \
            perm_calc_kr(item, dist, kr_params, perm_params, num_wells)

        array_coord_fit = np.array([array_coord[item, :, 0] - array_wells[item, 0],
                                    array_coord[item, :, 1] - array_wells[item, 1]])

        result_fit = curve_fit(approx_neighbour_perm, array_coord_fit, array_perm_add[item])[0]

        if plot_flag and item == 0:
            plot_perm_field(item, array_perm_total, array_perm_add, array_perm_well, result_fit, array_coord_fit)

        if last_num + item < max_num:
            with open(f'data/synthetic_dataset/perm_inc_files/PERM_{last_num + item}.INC', 'w+') as file:
                file.write('PERMX\n')
                file.write(' '.join(map(str, np.round(array_perm_total[item], 3))) + '\n')
                file.write('/')

        array_kr_params[item, 8:] = result_fit

    return array_kr_params


def plot_perm_field(item, array_perm_total, array_perm_add, array_perm_well, result_fit, array_coord_fit):
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))

    x_plot_init = np.linspace(-250, 250, 32)
    x_plot = 0.5 * (x_plot_init[1:] + x_plot_init[:-1])
    x, y = np.meshgrid(x_plot, x_plot)

    z = griddata((array_coord_fit[0], array_coord_fit[1]), array_perm_total[item], (x, y), method='linear')

    im = ax[0, 0].pcolormesh(x, y, z, cmap='rainbow')
    fig.colorbar(im, ax=ax[0, 0])
    ax[0, 0].set_title(r'$k, \mathrm{mD}$', fontsize=15)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=13)
    ax[0, 0].set_xlabel('x, m', fontsize=15)
    ax[0, 0].set_ylabel('y, m', fontsize=15)
    ax[0, 0].set_aspect('equal')

    z = griddata((array_coord_fit[0], array_coord_fit[1]), array_perm_add[item], (x, y), method='linear')
    max_k, min_k = max(array_perm_add[item]), min(array_perm_add[item])

    im = ax[0, 1].pcolormesh(x, y, z, cmap='rainbow', vmin=min_k, vmax=max_k)
    fig.colorbar(im, ax=ax[0, 1])
    ax[0, 1].set_title(r'$k^{\mathrm{neigh}}, \mathrm{mD}$', fontsize=15)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=13)
    ax[0, 1].set_xlabel('x, m', fontsize=15)
    ax[0, 1].set_ylabel('y, m', fontsize=15)
    ax[0, 1].set_aspect('equal')

    z = griddata((array_coord_fit[0], array_coord_fit[1]), array_perm_well[item], (x, y), method='linear')

    im = ax[1, 0].pcolormesh(x, y, z, cmap='rainbow')
    fig.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_title(r'$k^{\mathrm{well}}, \mathrm{mD}$', fontsize=15)
    ax[1, 0].tick_params(axis='both', which='major', labelsize=13)
    ax[1, 0].set_xlabel('x, m', fontsize=15)
    ax[1, 0].set_ylabel('y, m', fontsize=15)
    ax[1, 0].set_aspect('equal')

    z = griddata((array_coord_fit[0], array_coord_fit[1]),
                 approx_neighbour_perm(array_coord_fit, *result_fit), (x, y), method='linear')
    im = ax[1, 1].pcolormesh(x, y, z, cmap='rainbow', vmin=min_k, vmax=max_k)
    fig.colorbar(im, ax=ax[1, 1])
    ax[1, 1].set_title(r'$k^{\mathrm{neigh}} ~ \mathrm{approx.}, \mathrm{mD}$', fontsize=15)
    ax[1, 1].tick_params(axis='both', which='major', labelsize=13)
    ax[1, 1].set_xlabel('x, m', fontsize=15)
    ax[1, 1].set_ylabel('y, m', fontsize=15)
    ax[1, 1].set_aspect('equal')

    plt.tight_layout()

    plt.show()


def generate_perm(x_array, num_cells, max_samples):
    files = glob.glob('data/synthetic_dataset/perm_inc_files/*')
    for f in files:
        os.remove(f)

    num_samples_cur = 0
    kr_params = np.zeros((max_samples, 14))

    while num_samples_cur != max_samples - 1:
        min_dist = np.random.uniform(low=300, high=600.01)
        array_wells = generate_wells(x_array, min_dist)
        kr_params_cur = generate_perm_from_map(array_wells, x_array, num_cells, num_samples_cur + 1, max_samples)
        increment_samples = min(num_samples_cur + kr_params_cur.shape[0], max_samples - 1) - num_samples_cur
        kr_params[num_samples_cur: num_samples_cur + increment_samples] = kr_params_cur[:increment_samples]
        num_samples_cur += increment_samples

    np.save('data/synthetic_dataset/ann_input_data.npy', kr_params[:-1])
