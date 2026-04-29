import sys
from functools import total_ordering

from pipt.loop.assimilation import Assimilate
import pipt.misc_tools.analysis_tools as at
from input_output import read_config
from pipt import pipt_init
from ensemble.ensemble import Ensemble
from EMsim.EM import UTA2D
from prompt_toolkit.contrib.telnet import TelnetServer
from scipy.linalg import block_diag
from scipy.spatial.distance import cdist
from scipy.stats import norm
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from mako.runtime import Context
from mako.lookup import TemplateLookup
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import h5py
from ThreeDGiGEarth.common import h5_to_dict
import os
import matplotlib.pyplot as plt
import pickle
# fix the seed
import numpy as np
from geostat.gaussian_sim import fast_gaussian
import gstools as gs
import pandas as pd
from scipy.optimize import minimize

# remove folders in current directory starting with 'En_'
for folder in os.listdir('.'):
    if folder.startswith('En_') and os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)


reference_model_path = '/home/AD.NORCERESEARCH.NO/mlie/3DGiG/Jacobian/inversion/data/Benchmark-3/'
meter_to_feet = 3.28084

components = ['real(Bxx)', 'real(Bxy)', 'real(Bxz)',
                  'real(Byx)', 'real(Byy)', 'real(Byz)',
                  'real(Bzx)', 'real(Bzy)', 'real(Bzz)',
                  'img(Bxx)', 'img(Bxy)', 'img(Bxz)',
                  'img(Byx)', 'img(Byy)', 'img(Byz)',
                  'img(Bzx)', 'img(Bzy)', 'img(Bzz)']

data_type = 'Magnetic'


#T=np.loadtxt(input_dict['trajectory'],comments='%')
#TVD =T[:,1] MD = T[:,2] ED = T[:,5]

tot_assim_index = [[el] for el in range(250)]

indices_to_keep = [0, 2, 6, 8, 9, 11, 15, 17]

#extent of inversion area  setup_inversion_grid()
ext_x = 150 # feet
ded = 5# feet #--> initial  extent = 200 feet (1 m = 3.2808399 feet 1 foot = 0.3048 meters) 60 (120) m
ext_y = 10 # feet
ext_z = 150 # feet
dtvd = 0.43744#.. feet --> initial extent 150 feet

# data sensitive to the fault about 100 feet or 20 assimilation steps prior to the fault. Faults at 200, 600 and 1000 feet
# faults at logging points: 40, 120, and 200

# number of grid cells
nz = 15#101#457
numb_horizontal_grid = 5#16
dims = (numb_horizontal_grid, 1, nz)

param_size = np.prod(dims)

#type_kriging = 'ordinary'
type_kriging = 'universal'
#
type_kriging = 'none' # continue with final estimate from previous logging point

# inversion parameters
n_iter = int(4)
# initialize prior manually
ne = int(1)

#param_keys = ['rh','rv'] # fix the order
param_keys = ['rh','rh_ratio'] # fix the order

step_size = 1e-7
initial_step = 1e-7
decay_rate = 0.1

data = pd.read_pickle('data.pkl')
Cd = pd.read_pickle('var.pkl')
data_keys = list(Cd.columns)  # Extract column headers from Cd
tools = [str(dat) for dat in data_keys] # make into strings

pr = {}
Cm_ratio = {}

def vertical_spacing_log_points(logging_point_idx, TVD):
    dtvd = np.mean(np.diff(TVD[logging_point_idx:logging_point_idx+4]))
    return dtvd

def _no_logging_points_per_grid_cell_z(dZ, dtvd):
    no_log_points_per_cell_z = np.floor(dZ / dtvd)
    return no_log_points_per_cell_z

def _set_logging_point_state(logging_point_idx, TVD, MD, WX, WY):
    """Update tool/grid state for current logging point."""
    #UTA2D_sim.tool['tvd'] = TVD[logging_point_idx:logging_point_idx+1]
    well_x_m = WX[logging_point_idx]
    well_y_m = WY[logging_point_idx]
    well_z_m = TVD[logging_point_idx]
    UTA2D_sim.model['shift'] = {'x': well_x_m  - dX * (dims[0]//2),
                                'y': well_y_m  - dY * (dims[1]//2),
                                'z': well_z_m  - dZ * (dims[2]//2)}
    # only update shift in z if logging point has moved one whole grid cell.

    #UTA2D_sim.tool['MD'] = MD[logging_point_idx:logging_point_idx+1]
    #UTA2D_sim.tool['X'] = WX[logging_point_idx:logging_point_idx+1]
    #UTA2D_sim.tool['ijk'] = WY[logging_point_idx:logging_point_idx+1]
    #UTA2D_input['shift'] = UTA2D_sim.model['shift']

def setup_input_dict(reference_model_path, data_type):
    UTA_input_dict = {'toolflag': 0,
                      'datasign': 3 if data_type == 'UDAR' else 0,
                      'anisoflag': 1,
                      'toolsetting': f'{reference_model_path}/ascii/tool.inp',
                      'reference_model': f'{reference_model_path}/globalmodel.h5',
                      'parallel': 1,
                      'map': {'ratio': [1, 3]},
                      }

    with h5py.File(UTA_input_dict['reference_model'], "r") as f:
        ref_model = h5_to_dict(f)

    wellpath = ref_model['wellpath']
    TVD = wellpath['Z'][:, 0] * meter_to_feet  # in feet
    MD = wellpath['Distance'][:, 0] * meter_to_feet  # in feet
    WX = wellpath['X'][:, 0] * meter_to_feet  # in feet
    WY = wellpath['Y'][:, 0] * meter_to_feet  # in feet

    UTA2D_input = {key: UTA_input_dict[key] for key in ['toolflag', 'datasign', 'anisoflag', 'reference_model', 'map',
                                                        'toolsetting']}

    UTA2D_input['tvd'] = TVD
    UTA2D_input['MD'] = MD
    UTA2D_input['X'] = WX
    UTA2D_input['ijk'] = WY
    UTA2D_input['dip'] = wellpath['Inclination'][:,0]
    UTA2D_input['azim'] = wellpath['Azimuth'][:,0]
    #T = np.loadtxt(input_dict['trajectory'], comments='%')
    #    self.tool['tvd'] =T[:,1]
    #    self.tool['MD']  =T[:,2]
    #    self.tool['dip'] =T[:,3]
    #    self.tool['Azim'] =T[:,4]
    #    self.tool['X']   =T[:,5]
    #    self.tool['nlog']=len(self.tool['tvd'])
    #    try:
    #        self.tool['ijk'] = T[:,6] # get tool ijk elem
    # TVD =T[:,1] MD = T[:,2] ED = T[:,5]

    UTA2D_input['dims'] = dims
    UTA2D_input['dX'] = dX
    UTA2D_input['dY'] = dY
    UTA2D_input['dZ'] = dZ
    UTA2D_input['shift'] = {'x': 0,
                                'y': 0,
                                'z': 0}
    # UTA2D_input['surface_depth'] = grid_boundaries

    UTA2D_input['jacobian'] = True

    return UTA2D_input, TVD, MD, WX, WY

def setup_simulators(UTA2D_input, logging_point_idx, TVD, MD, WX, WY, shift_cells):
    """
    Set up the UTA2D and UTA3D simulators with the given reference model path and data type.
    """
    global UTA2D_sim #, TVD, MD, WX, WY, wellpath

    sim_info = {
    'obsname': 'tvd',
    'assimindex': [[assim_index[0]]],
    'datatype': tools
    }



    UTA2D_sim = UTA2D({**UTA2D_input, **sim_info})
    if shift_cells:
        _set_logging_point_state(logging_point_idx, TVD, MD, WX, WY)

    UTA2D_sim.setup_fwd_run(redund_sim=None)

def _build_param_dict(param, min_ratio = 1, max_ratio = 3):
    """Map optimizer parameters to simulator input dictionary."""
    n_params_per_type = dims[0] * dims[2]  # nx * nz
    # Optimizer vector uses (nx, nz) with C-order flattening (z-fastest).
    rh_2d = param[:n_params_per_type].reshape(dims[0], dims[2], order='C')
    rh_ratio_2d = param[n_params_per_type:].reshape(dims[0], dims[2], order='C')

    # Broadcast 2D arrays to 3D by tiling along the y-axis
    rh = np.tile(rh_2d[:, np.newaxis, :], (1, dims[1], 1))

    # make rv
    rh_ratio = np.tile(rh_ratio_2d[:, np.newaxis, :], (1, dims[1], 1))
    rv = np.log(np.exp(rh) * (norm.cdf(rh_ratio) * (max_ratio - min_ratio) + min_ratio))

    param_dict = {'rh': rh.flatten(order='F'),
                  'rv': rv.flatten(order='F')
                  }
    return param_dict, rh_2d, rh_ratio_2d

def find_overlap_region_kriging(nz, nx, dz, dx):#ded, dtvd):

    z = 0 + dz * np.arange(nz, dtype=float)
    x = 0 + dx * np.arange(nx, dtype=float)
    # area of overlapping inversion domains is in between logging points
    nx_overlap = nx#int(np.floor(nx * 5 / 6))  # int(np.ceil(numb_horizontal_grid / 2))
    if shift_cells:
        nz_not_overlap = 1 # how many grid cells inversion domain is shifted downwards
    else:
        nz_not_overlap = 0#int(np.floor(dtvd / dz))
    x_overlap = x[:nx_overlap]
    z_overlap = z[nz_not_overlap:]
    # the ordering of the param_vec is (gridsize +1, numb_horizontal_grid)
    x_grid_overlap, z_grid_overlap = np.meshgrid(x_overlap, z_overlap, indexing="xy")
    #z_grid_overlap, x_grid_overlap = np.meshgrid(z_overlap, x_overlap, indexing="ij")
    x_coord_overlap = np.array(x_grid_overlap).ravel()
    z_coord_overlap = np.array(z_grid_overlap).ravel()
    #x_grid, z_grid = np.meshgrid(x, z, indexing="xy")
    z += dz


    return x, z, nx_overlap, nz_not_overlap, x_coord_overlap, z_coord_overlap

def kriging(p_input, x, z, key, nx_overlap, nz_not_overlap, x_coord_overlap, z_coord_overlap, type_kriging = 'ordinary', red_fact = 0.75, corr_length_z = 40, anisotropy_factor = 50, debug_on = False):

    # type_kriging = 'ordinary'
    # red_fact: # factor for reduction in correlation length during kriging testing
    #param_size = len(z) * len(x)
    #Sill: Represents the variance at larger distances. If your errors are significantly larger, it may indicate the need for a higher sill.

    #Range: Indicates the distance over which points are correlated. A short range might suggest that the correlation
    #       diminishes very quickly, while a long range means that points are still correlated even when far apart.

    #Nugget: Represents the measurement error and micro-scale variability. A higher nugget suggests increased noise in your data.

    obs = p_input[nz_not_overlap:, :nx_overlap].flatten(order = 'C')
    if key == 'rh_ratio_test':
        obs = norm.cdf(obs) * (3 - 1) + 1

    # only perform Kriging if field is not a constant
    if p_input.max() - p_input.min() < 0.0001 or type_kriging == 'none':
        if nz_not_overlap == 0:
            log_mean = p_input
        else:
            # Omit the first n rows
            remaining_rows = p_input[nz_not_overlap:]
            last_row = remaining_rows[-1]
            duplicated_rows = np.tile(last_row, (nz_not_overlap, 1))
            log_mean = np.vstack((remaining_rows, duplicated_rows))

        log_std = 1
    else:
        # Perform kriging: get conditional mean and variance on the grid
        success_kriging = False


        n_trials = 0
        while not success_kriging:
            # estimate range (correlation length) by Leave-One-Out Cross-Validation:
            variogram_parameters = {"sill": 1, "range": corr_length_z, "nugget": 0.05}

            match type_kriging:
                case "universal":
                    # Perform kriging: get conditional mean and variance on the grid
                    # higher "sill" improve correlation through affecting variance away from observation
                    # higher nugget increase uncertainty in output
                    variogram_model = "spherical"# variogram_model="gaussian" gaussian is more smooth
                    n_lags = 20
                    OK = UniversalKriging(x_coord_overlap / anisotropy_factor, z_coord_overlap, obs, exact_values=False,
                                        variogram_model= variogram_model,
                                        variogram_parameters=variogram_parameters,
                                        nlags=n_lags,
                                        enable_plotting=False, verbose=False)
                    # OK = UniversalKriging(x_coord_overlap, z_coord_overlap, obs)
                case _:
                    # increase dx relative to dz to reduce correlation lengths in the x-direction
                    # covarance model (get parameters from ensemble?
                    #cov_model = gs.Gaussian(dim = 2, len_scale=10, anis=0.02, angles=0.0, nugget=0.5)
                    #OK = OrdinaryKriging(x_coord_overlap * red_fact_dx, z_coord_overlap, obs)  # , cov_model, exact_values=True)
                    OK = OrdinaryKriging(x_coord_overlap / anisotropy_factor, z_coord_overlap, obs, variogram_model="spherical",
                                         variogram_parameters=variogram_parameters, enable_plotting=False, verbose=False)

            log_mean, log_var = OK.execute("grid", x / anisotropy_factor, z)

            n_trials += 1
            if np.linalg.norm(log_mean - p_input, ord=2) / np.sqrt(np.size(log_mean)) < 0.2:

                success_kriging = True
            else:
                if n_trials > 4:
                    #red_fact += 0.05
                    #corr_length_z *= red_fact
                    #n_trials = 0
                    success_kriging = True
                elif n_trials <= 2:
                    corr_length_z *= red_fact
                else:
                    anisotropy_factor *= red_fact
            r_squared = r2_score(p_input, log_mean)
            mse_error = mean_squared_error(p_input, log_mean)
            print(r_squared, mse_error)


        # Extract the fitted variogram parameters, particularly the 'range'
        fitted_variogram_model = OK.variogram_model_parameters
        correlation_length = fitted_variogram_model[1]
        print(f"Estimated vert. correlation length (feet): {correlation_length} vs input to kriging {corr_length_z} redfact {red_fact}")
        corr_length_z =  correlation_length
        corr_length_x = corr_length_z * anisotropy_factor

        # Generate correlated noise
        #x_grid, z_grid = np.meshgrid(x /20, z)
        #grid_points = np.vstack((x_grid.ravel(), z_grid.ravel())).T
        #correlated_noise = generate_correlated_field(grid_points, variogram_parameters["sill"],
                                                     #variogram_parameters["range"], variogram_parameters["nugget"])


        if debug_on:
            plt.figure()
            if key == 'rh' or key == 'rv':
                plt.imshow(np.exp(p_input[nz_not_overlap:, :nx_overlap]), aspect='auto',
                           extent=[x[0], x[nx_overlap - 1], z[nz_not_overlap], z[-1]])
            else:
                value = norm.cdf(p_input[nz_not_overlap:, :nx_overlap]) * (3-1) + 1#(input_dict['map']['ratio'][1] - input_dict['map']['ratio'][0]) + input_dict['map']['ratio'][0]
                plt.imshow(value, aspect='auto',
                           extent=[x[0], x[nx_overlap - 1], z[nz_not_overlap], z[-1]])
            plt.colorbar()
            plt.title(f'Test - {key} before kriging' + str(len(z)))
            plt.savefig(f'Test - {key} before kriging' + str(len(z)) + '.png')

            plt.figure()
            if key == 'rh' or key == 'rv':
                plt.imshow(np.exp(log_mean), aspect='auto', extent=[x[0], x[-1], z[0], z[-1]])
            else:
                value = norm.cdf(log_mean) * (3-1) + 1
                plt.imshow(value, aspect='auto', extent=[x[0], x[-1], z[0], z[-1]])

            plt.colorbar()
            plt.title(f'Test - {key} after kriging' + str(len(z)))
            plt.savefig(f'Test - {key} after kriging' + str(len(z)) + '.png')

        log_var_arr = np.asarray(log_var)
        log_var_arr = np.where(log_var_arr < 0, 0.01, log_var_arr)
        log_std = np.sqrt(log_var_arr)
    #print(f"Estimated log parameter uncertainty after kriging: {np.mean(log_std)}")

    return log_mean, log_std, corr_length_z

def setup_inversion_grid(extent_x, extent_y, extent_z):

    # ed: x-coordinate of logging point?
    # ded: spacing in x direction between logging points
    # tvd: depth of active logging points
    # dtvd: difference in depth between different logging points (used to define dz)

    # allow for some overlap of inversion domains between successive logging points
    #dx = ded / numb_horizontal_grid * (3/2)
    dx = extent_x / dims[0]
    # make sure the tool is in the center of the grid
    #x_min = ed - dx * numb_horizontal_grid / 2
    # inversion domain dz*grid_size approximately 200 feet high
    # z_fact = np.floor( 200 / (grid_size * dtvd) ) not sure if it is important that dz is an integer value of the height change between logging points
    #dz = dtvd*z_fact
    dz = extent_z / dims[2]
    dy = extent_y / dims[1] #feet

    # set the grid boundaries in z-direction such that TVD[0] is in the center of the inversion domain, and gives the reference to the boundaries
    # grid_boundaries = [tvd+(dz/2) - dz*np.ceil(grid_size/2) + dz*i for i in range(grid_size+1)]
    # grid_boundaries = [tvd + (dz/2) - dz * np.ceil(grid_size / 2) + dz * i for i in range(grid_size)]
    # plt.figure(figsize=(6, 10))
    # for boundary in grid_boundaries:
    #     plt.axhline(y=boundary, color='r', linestyle='--')
    #     plt.plot(0,boundary, 'rx')  # Add 'x' style at the boundary position
    #
    # plt.plot(0,TVD[0], 'ro')
    # plt.plot(0,TVD[10], 'co')
    # plt.xlabel('Tool Position')
    # plt.ylabel('Depth')
    # plt.title('Boundaries')
    # plt.gca().invert_yaxis()
    # plt.grid(True)
    # plt.savefig('boundaries.png')

    return dx, dy, dz #, x_min, grid_boundaries# input_dict

def generate_correlated_field(grid_points, sill, range_, nugget):
    # Compute pairwise distances
    dists = cdist(grid_points, grid_points)
    # Compute covariance matrix based on spherical model
    cov_matrix = sill * (1.5 * (dists / range_) - 0.5 * (dists / range_)**3) * (dists < range_) + (dists >= range_) * sill + np.eye(len(grid_points)) * nugget
    # Generate random values
    random_values = np.random.multivariate_normal(np.zeros(len(grid_points)), cov_matrix)
    return random_values

def set_hard_contraints_param_values(key):
    # Limits on parameter values
    if key == 'rh' or key == 'rv':
        min_val = -4
        max_val = 10
    elif key == 'rh_ratio':
        min_val = -10
        max_val = 10
    else:
        min_val = -10
        max_val = 10
        print('mean not defined for', {key})

    return min_val, max_val

def gen_prior(postr, nz, nx, dz, dx, ded, dtvd, uncert_inflation = 0, comb_ge_kriging = False, use_ensemble_member = False, first_assimilation_step = False, h_corr = 300, v_corr = 40, red_fact = 0.75, min_ratio = 1, max_ratio = 3, shift_cells = False):
    # Allow to merge field
    # Alt 1, invert for rh and rv


    param_offset = 0
    v_corr = 50
    anisotropy_factor = 50
    h_corr = v_corr * anisotropy_factor

    for key in param_keys:

        log_std = [0.3, 0.8]  # np.mean([abs(log_mean), abs(log_mean_rest)])
        # increase parameter uncertainty if initial ensemble predictions do not cover the data
        log_std[1] = log_std[1]*(1+uncert_inflation)

        # specify uncertainty/prior ensemble spread


        #Alt A: data coverage is OK--use
        if first_assimilation_step:
            min_val, max_val = set_hard_contraints_param_values(key)
            pr[f"{key}_min"] = min_val
            pr[f"{key}_max"] = max_val

            log_mean, log_std, prior_param = gen_prior_fast_gaussian(key, nz, nx, dz, dx, np.max(log_std), first_assimilation_step, h_corr, v_corr, min_ratio, max_ratio)

        else:
            log_mean, log_std, prior_param = gen_prior_kriging(postr, key, param_offset, nz, nx, dz, dx, ded, dtvd, log_std, use_ensemble_member, 0.75, v_corr, anisotropy_factor, shift_cells)

            if comb_ge_kriging: # combine kriging with gausssian prior
                new_cells = nx - 1
                log_mean_gauss, log_std_gauss, param_gauss = gen_prior_fast_gaussian(key, nz, nx, dz, dx, np.max(log_std), first_assimilation_step, h_corr, v_corr)
                log_mean2D = log_mean.reshape(nz, nx, order='C')
                log_mean_gauss2D = log_mean_gauss.reshape(nz, nx, order='C')
                log_mean2D[:, new_cells:] = log_mean_gauss2D[:, new_cells:]
                log_mean = log_mean2D.flatten(order='C')

                for ind_ne in range(ne):
                    param2D = prior_param[:,ind_ne].reshape(nz, nx, order='C')
                    param2D_gauss = param_gauss[:, ind_ne].reshape(nz, nx, order='C')
                    param2D[:, new_cells:] = param2D_gauss[:, new_cells:]
                    prior_param[:,ind_ne] = param2D.flatten(order='C')

        param_offset += nx*nz

        pr[f"{key}"] = prior_param
        pr[f"{key}_std"] = log_std
        pr[f"{key}_mean"] = log_mean


        # Create full block diagonal covariance matrix Cm
        # Block 1: Cm_rh_ratio for 'rh' parameters
        # Block 2: Cm_rh_ratio for 'rh_ratio' parameters
        Cm_ratio[f"{key}"] = np.eye(param_size) * log_std ** 2

    Cm = np.diag(block_diag(*(Cm_ratio[key] for key in param_keys)))

    return pr, Cm

def gen_prior_fast_gaussian(key, nz, nx, dz, dx, log_std, first_assimilation_step = True, h_corr = 300, v_corr = 40, min_ratio = 1, max_ratio = 3):
    # horizontal_skin_depth = 20  # feet
    # correlation lengths:
    #       h_corr = 300  # ft
    #       v_corr = 40  # ft
    # Alt 1 update_mean = True: update mean (and std dev) based on results from previous logging point
    np.random.seed(10)
    param_dim = nz * nx

    h_corr_cells = int(np.ceil(h_corr / dx))  # number of cells to correlate in the vertical direction
    v_corr_cells = int(np.ceil(v_corr / dz))

    # specify ensemble mean values and spread around mean for first assimilation step
    if first_assimilation_step:
        if key == 'rh':
            log_mean = np.log(1.5)
            log_mean_rest = np.log(30)
        elif key == 'rh_ratio':
            log_mean = norm.ppf((4 / 1.5 - min_ratio) / (
                        max_ratio - min_ratio))
            log_mean_rest = norm.ppf((1.01 - min_ratio) / (
                    max_ratio - min_ratio))
            # print('rv: ',1.5 * (norm.cdf(log_mean) * (input_dict['map']['ratio'][1] - input_dict['map']['ratio'][0]) + input_dict['map']['ratio'][0]))
        elif key == 'rv':
            log_mean = np.log(4)
            log_mean_rest = np.log(30)
        else:
            log_mean = 1
            log_mean_rest = log_mean
            print('mean not defined for', {key})

    else:  # extract one average value for each parameter
        if key == 'rh':
            # log_mean = np.mean([v[:param_size] for v in param_vec_ens]) # makes one value averaged over inversion domain
            # log_mean = np.mean(np.array([v[:param_size] for v in param_vec_ens]),0) # averages only over ensemble members not all model parameters
            log_mean = np.average(np.array([v[:param_dim] for v in param_vec_ens]),
                                  0, 1 / tot_loss_ens[
                                      :, -1])  # same as above, but gives the least weight to those with large misfit
        elif key == 'rh_ratio':
            # log_mean = np.mean([v[param_size:] for v in param_vec_ens])
            log_mean = np.average(np.array([v[param_dim:] for v in param_vec_ens]),
                                  0, 1 / tot_loss_ens[:, -1])
        elif key == 'rv':
            # log_mean = np.mean([v[param_size:] for v in param_vec_ens])
            log_mean = np.average(np.array([v[param_dim:] for v in param_vec_ens]),
                                  0, 1 / tot_loss_ens[:, -1])
        else:
            log_mean = 1
            print('mean not defined for', {key})
        log_mean_rest = log_mean

    # Alt 1. generate ensemble of parameter values based on mean values and parameter uncertainties
    if first_assimilation_step:
        block_A = False
    else:
        block_A = False
    if block_A:
        n_cells1 = int(round((17000 - int(UTA2D_sim['shift']['z'])) / dz))
        n_cells2 = int(round((70 / dz)))
        n_cells3 = nz - n_cells1 - n_cells2
        segment1 = np.full((n_cells1, 1), log_mean)
        segment2 = np.full((n_cells2, 1), log_mean_rest)
        segment3 = np.full((n_cells3, 1), log_mean)
        vect = np.vstack((segment1, segment2, segment3))
        prior_param = np.repeat(
            np.tile(vect.reshape(-1, 1), nx).flatten(order='C')[
                :, np.newaxis], ne, axis=1) + np.hstack([fast_gaussian(np.array([nx, nz]), np.array([log_std]),
                                                                       np.array([h_corr_cells, v_corr_cells])) for _
                                                         in range(ne)])
    else:
        # homogeneous prior
        if np.isscalar(log_mean) == 1:
            prior_param = np.repeat(
                np.tile((np.ones((nz)) * log_mean).reshape(-1, 1), nx).flatten(order='C')[
                    :, np.newaxis], ne, axis=1) + np.hstack([fast_gaussian(np.array([nx, nz]), np.array([log_std]),
                                                                           np.array([h_corr_cells, v_corr_cells]))
                                                             for _ in range(ne)])
        else:  # perturb ensemble mean
            prior_param = np.tile(np.array(log_mean / log_mean.max())[:, np.newaxis], ne) + np.hstack(
                [fast_gaussian(np.array([nx, nz]), np.array([np.max(log_std)]),
                               np.array([h_corr_cells, v_corr_cells])) for _ in range(ne)])

    #
    prior_param = np.stack(prior_param, axis=0)


    # prior_param = np.clip(prior_param, min_val, max_val)


    return log_mean, log_std, prior_param

def gen_prior_kriging(postr, key, param_offset, nz, nx, dz, dx, ded, dtvd, log_std, use_ensemble_member, red_fact = 0.75, v_corr = 40, anisotropy_factor= 50, shift_cells = False):
    # horizontal_skin_depth = 20  # feet
    # correlation lengths:
    #       h_corr = 300  # ft
    #       v_corr = 40  # ft
    # Alt 1 update_mean = True: update mean (and std dev) based on results from previous logging point
    # Alt 2 update_mean = False: condition prior ensemble to  a subset of the results from the previous logging point using Kriging
    # Alt 1.1. use_ensemble_member = False: use ensemble mean as input for kriging
    # Alt 1.2. use_ensemble_member = true: use each ensemble individually as input for kriging

    np.random.seed(10)
    param_dim = nz * nx

    x, z, nx_overlap, nz_not_overlap, x_coord_overlap, z_coord_overlap = find_overlap_region_kriging(nz, nx, dz, dx)#ded, dtvd)

    # generate ensemble of parameter values based on kriging
    prior_param=[]
    post_param = []
    # Alt 1: base kriging on ensemble mean
    if not use_ensemble_member:
        param_vec_ens_array = np.vstack(param_vec_ens)  # shape (n_ens, n_param)
        #param_vec_mean = np.mean(param_vec_ens_array, axis=0)  # 1D array, length
        param_vec_mean = np.average(param_vec_ens_array,
                              0, 1 / tot_loss_ens[:, -1])
        param2d = np.reshape(param_vec_mean[param_offset:param_offset + param_dim], (len(z), len(x)))

        log_mean, log_std_kriging, v_corr = kriging(param2d, x, z, key,
                                                            nx_overlap, nz_not_overlap, x_coord_overlap,
                                                            z_coord_overlap, type_kriging, red_fact, v_corr, anisotropy_factor)

        for i in range(ne):
            noise = np.random.randn(1, )  # N(0, 1) at each cell
            log_std_kriging *= log_std / np.max(log_std_kriging)
            prior_param.append(np.array(log_mean).ravel()  + log_std_kriging.ravel() * noise)
        # log_std = np.mean(log_std_kriging)
        log_std = log_std_kriging


    # Alt 2: generate prior ensemble based on individual ensemble members
    else:
        st_dev_ens = []

        for ne_count in range(ne):
            param2d = np.reshape(postr[ne_count][param_offset:param_offset + param_dim], (len(z), len(x)))
            if key == 'rh':
                post_param_i = postr[ne_count][param_offset:param_offset + param_dim]
                post_param.append(np.array(post_param_i).ravel())
            log_mean_i, log_std_i, v_corr = kriging(param2d, x, z, key,
                                                            nx_overlap, nz_not_overlap, x_coord_overlap,
                                                            z_coord_overlap, type_kriging, red_fact, v_corr, anisotropy_factor)

            prior_param.append(np.array(log_mean_i).ravel())
            log_std_i = log_std_i / np.max(log_std_i) * np.min(log_std)
            st_dev_ens.append(np.array(log_std_i).ravel())
            if key == 'rh':
                print(f'max change in rh in gen_prior.py after kriging per ensemble: {np.max(np.exp(log_mean_i) - np.exp(param2d)):0.2f}')
        # ensemble mean values
        #log_mean = np.mean([v[:param_dim] for v in param])
        log_mean = np.average(np.array([v[:param_dim] for v in prior_param]),
                              0, 1 / tot_loss_ens[:, -1])
        #log_std = np.mean(np.array([v[:param_dim] for v in st_dev_ens]))
        log_std = np.average(np.array([v[:param_dim] for v in st_dev_ens]),
                              0, 1 / tot_loss_ens[:, -1])


    prior_param = np.stack(prior_param, axis = 1)
    if key == 'rh':
        post_param = np.stack(post_param, axis=1)



    # add parameter uncertainty to kriging results
    if key == 'rh':
        print(f'max change in rh in gen_prior.py after kriging: {np.max(np.exp(prior_param)-np.exp(post_param)):0.2f}')

    log_mean = np.reshape(log_mean, (param_size,))
    log_std = log_std.flatten()


    return log_mean, log_std, prior_param

def custom_loss(predictions, theta, pr_theta):
    # Example: weighted mean squared error
    residuals_data = data_real - predictions.flatten()
    data_loss = 0.5 * np.sum((residuals_data ** 2) / Cd_vec)

    theta_prior = np.ones_like(theta)
    offset = 0
    for key in param_keys:  # or for key in sorted(pr_theta) if you want fixed order
        theta_prior[offset:offset + param_size] *= pr_theta[f"{key}_mean"]
        offset += param_size #+ numb_horizontal_grid
    #theta_prior[:param_size] = theta_prior[:param_size] * pr_theta['rh_mean']
    #theta_prior[param_size:] = theta_prior[param_size:] * pr_theta['rv_mean']

    residuals_theta = theta - theta_prior

    theta_loss = 0.5 * np.sum((residuals_theta**2) / Cm)

    loss = data_loss #+ theta_loss

    return loss, data_loss, theta_loss

def map_update_data_space(theta, theta_prior, f_theta, J, y, Cd_diag, Cm):
    """
    Gauss-Newton MAP update in data-space using matrix inversion lemma for the loss function:
    L(theta) = 0.5 * ||y - f(theta)||^2_{Cd^-1} + 0.5 * ||theta - theta_prior||^2_{Cm^-1}

    Using Sherman-Morrison-Woodbury formula to avoid inverting [n x n] matrix.
    Instead, we invert [d x d] matrix.

    Parameters:
    theta: [n] - current parameter estimate
    f_theta: [d] - forward model prediction at current theta
    J: [d, n] - Jacobian matrix
    y: [d] - observed data
    Cd_diag: [d] - diagonal of data covariance matrix

    Returns:
    theta_new: [n] - updated parameter estimate
    """

    theta = theta.flatten()
    #theta_prior = np.ones_like(theta)*log_rh_mean

    # Residuals
    r = f_theta.flatten() - y  # [d]
    r_m = theta - theta_prior  # [n]

    # Compute H_data = Cd + J @ Cm @ J.T more efficiently
    # Avoid forming full matrix when possible
    CM_JT = Cm[:, np.newaxis] * J.T  # [n, d] - element-wise multiplication when Cm is diagonal vector
    H_data = np.diag(Cd_diag) + J @ CM_JT  # [d, d]
    rh = r - J @ r_m  # [d]

    w = np.linalg.solve(H_data, rh)  # [d]

    delta_theta = -r_m - CM_JT @ w  #

    theta_new = theta + delta_theta

    return theta_new

def map_update_data_space_LM(theta, pr_theta, f_theta, J, y, Cd_diag, step_lambda, Cm):
    """
    Gauss-Newton MAP update in data-space using matrix inversion lemma for the loss function:
    L(theta) = 0.5 * ||y - f(theta)||^2_{Cd^-1} + 0.5 * ||theta - theta_prior||^2_{Cm^-1}

    Using Sherman-Morrison-Woodbury formula to avoid inverting [n x n] matrix.
    Instead, we invert [d x d] matrix.

    Parameters:
    theta: [n] - current parameter estimate
    f_theta: [d] - forward model prediction at current theta
    J: [d, n] - Jacobian matrix
    y: [d] - observed data
    Cd_diag: [d] - diagonal of data covariance matrix

    Returns:
    theta_new: [n] - updated parameter estimate
    """

    theta = theta.flatten()

    theta_prior = np.ones_like(theta)
    offset = 0
    for key in param_keys:  # or for key in sorted(pr_theta) if you want fixed order
        theta_prior[offset:offset + param_size] *= pr_theta[f"{key}_mean"]
        offset += param_size #+ numb_horizontal_grid

    # Residuals
    r = f_theta.flatten() - y  # [d]
    r_m = (theta - theta_prior)/(1 + step_lambda)  # [n]

    # Compute H_data = Cd + J @ Cm @ J.T more efficiently
    # Avoid forming full matrix when possible
    CM_JT = Cm[:, np.newaxis] * J.T  # [n, d] - element-wise multiplication when Cm is diagonal vector
    H_data = (1+step_lambda)*np.diag(Cd_diag) + J @ CM_JT  # [d, d]
    rh = r - J @ r_m  # [d]

    w = np.linalg.solve(H_data, rh)  # [d]

    # Step 5: Gauss-Newton update: delta_theta = Cm g - Cm J^T v
    delta_theta = -r_m - CM_JT @ w  #

    theta_new = theta + delta_theta
    offset = 0
    for key in param_keys:  # or for key in sorted(pr_theta) if you want fixed order
        theta_new[offset:offset + param_size] = np.clip(theta_new[offset:offset + param_size], pr_theta[f"{key}_min"],pr_theta[f"{key}_max"])
        offset += param_size #+ numb_horizontal_grid

    return theta_new

def has_converged(tot_loss, data_loss, param_loss, jacobian_matrix, step_lambda, tol_rel=0.001):
    """
    Return success if convergence is met after following criteria:
     1. last loss is NOT at least `tol_rel` (default 1%) lower than previous.
    Needs at least 2 entries in tot_loss.
    """

    if len(tot_loss) < 2:
        conv = False
        return conv  # or raise an error

    prev = tot_loss[-2]
    curr = tot_loss[-1]

    # Convergence check: Relative step size of data misfit or state change less than tolerance
    # Improvement (positive if curr < prev)
    improvement = prev - curr

    # Required improvement
    required = tol_rel * prev
    conv_improvement = improvement < required



    # test if noise level in the data is reached:
    n_data_points = np.size(jacobian_matrix, axis = 0)
    n_parameters = np.size(jacobian_matrix, axis=1)

    # test if step_lambda is large (early in iterations) or small (close to a minimum)
    if conv_improvement:
        if step_lambda*5 > 0.8 * (curr / (2 * n_data_points)):
            # try another update with smaller lambda
            conv_improvement = False

    conv_data_noise = data_loss < n_data_points
    conv_param = param_loss < n_parameters

    condition_number = np.linalg.cond(jacobian_matrix)
    jacobian_norm = np.linalg.norm(jacobian_matrix)

    # check if any convergence criteria are fulfilled
    if any([conv_improvement, conv_data_noise]):
        print(f"conv_improvement, conv_data_noise, conv_param: {[conv_improvement, conv_data_noise, conv_param]}")
        conv = True
    else:
        conv = False

    why_stop = {'misfit_stop': conv_improvement,
               'misfit': curr,
               'prev_misfit': prev,
               'Jacobian norm': jacobian_norm,
               'Jacobian condition number': condition_number}


    # If improvement < required, convergence is met
    return conv

def construct_full_jacobian(min_ratio = 1, max_ratio = 3):
    jac_list = []
    jac_list_full = []
    offset = 0
    for param_idx, param_key in enumerate(param_keys):
        # Collect Jacobian for this parameter across all tool settings in the same order as data_keys
        jac_param_list = []
        jac_param_list_full = []
        for data_key in tools:
            # jacobian[0][data_key][param_idx] has shape (num_data, num_k, num_j, num_i)
            # Reshape to (num_data, num_k*num_j*num_i)
            jac_tool = jacobian[assim_index[0]][data_key][param_idx].reshape(jacobian[assim_index[0]][data_key][param_idx].shape[0], -1)
            jac_tool_keep = jac_tool[indices_to_keep,]
            jac_param_list.append(jac_tool_keep)
            jac_param_list_full.append(jac_tool)

        jac_param = np.vstack(jac_param_list)
        jac_param_full = np.vstack(jac_param_list_full)
        # no 2 Jacobian is wrt r_h, we invert for the parameter is p = log(r_h). Add term in Jacobian reflecting d r_h/ d p = exp(p)
        # drh_dparam = np.exp(param_vec[np.prod(input_dict['dims']) * param_idx:np.prod(input_dict['dims']) * (param_idx + 1),])
        if param_key == 'rh_ratio':
            dv_dp = norm.pdf(param_vec[offset:offset + param_size]) * (max_ratio - min_ratio)
            jac_param = jac_param * dv_dp
            jac_param_full = jac_param_full * dv_dp

        offset += param_size #+ numb_horizontal_grid


        # no 3
        # jac_param = np.flipud(jac_param)
        # jac_1D = jac_param.flatten()
        # mean_jac_3D = np.reshape(jac_1D, (108, input_dict['dims'][0], input_dict['dims'][2]), order="F")
        # jac_param = np.reshape(mean_jac_3D, (108, np.prod(input_dict['dims'])), order='C')
        jac_list.append(jac_param)
        jac_list_full.append(jac_param_full)

    J = np.hstack(jac_list)
    J_full = np.hstack(jac_list_full)
    return J, J_full

def plot_coverage(pred_ens_prior):
    start_index = 0
    for dat_ind, dat in enumerate(data_keys):
        dat_key = f"('{dat[0]}', '{dat[1]}')"
        n_data_points = len(data.iloc[assim_index[0]][dat])  # Number of data points
        x = np.arange(n_data_points)
        plt.figure()
        for count_ind, comp_ind in enumerate(indices_to_keep):
            data_point_assessed = data.iloc[assim_index[0]][dat][comp_ind]
            max_value = pred_ens_prior[0][comp_ind+start_index]
            min_value = pred_ens_prior[0][comp_ind+start_index]
            for ind_ne in range(ne):
                plt.plot(x[comp_ind], pred_ens_prior[ind_ne][comp_ind+start_index], marker='o', linestyle='None',c='0.35')
                max_value = max(max_value, pred_ens_prior[ind_ne][comp_ind+start_index])
                min_value = min(min_value,
                                pred_ens_prior[ind_ne][comp_ind + start_index])

            plt.plot(x[comp_ind], data_point_assessed, 'g*')

            noise_in_data_point_assessed = Cd_std[count_ind + dat_ind*len(indices_to_keep)]
            max_value += noise_in_data_point_assessed
            min_value -= noise_in_data_point_assessed
            if max_value <  data_point_assessed:#cover_high[comp_ind]:
                plt.plot(x[comp_ind], data_point_assessed, 'r*')
            if min_value >  data_point_assessed:#cover_low[comp_ind]:
                plt.plot(x[comp_ind], data_point_assessed, 'r*')

        start_index += n_data_points

        selected_components = [components[i] for i in
                               indices_to_keep]
        plt.ylabel(f'{dat_key}')
        plt.xlabel(", ".join(selected_components))
        plt.title(f'logging no. {assim_index[0]}')

    plt.show()
        # plt.savefig(self.folder + typ.replace(' ', '_'))
        # plt.close()

def check_coverage(call_plot_coverage = False):
    pred_ens_prior = []
    for member in range(ne):
        param = {k: pr[k][:, member] for k in param_keys}
        param_vec = np.concatenate([param[key].flatten() for key in param_keys])
        param_dict = {}
        offset = 0
        for key in param_keys:
            param_dict[key] = param_vec[offset:offset + param_size]
            offset += param_size #+ numb_horizontal_grid

        if isinstance(param_vec, (bool, np.bool_)):
            print('something went wrong')

        pred_prior, jacobian = UTA2D_sim.run_fwd_sim(param_dict, 0)
        pred_vec_prior = np.concatenate([pred_prior[assim_index[0]][k] for k in tools])
        pred_ens_prior.append(pred_vec_prior)

    success_coverage = True
    # Dimensions
    num_data_types = len(data_keys)
    num_components = len(indices_to_keep)

    coverage = np.zeros(num_data_types * num_components, dtype=bool)
    start_index = 0
    for dat_ind, dat in enumerate(data_keys):
        n_data_points = len(data.iloc[assim_index[0]][dat])  # Number of data points for each tool settting
        for count_ind, comp_ind in enumerate(indices_to_keep):
            data_point_assessed = data.iloc[assim_index[0]][dat][comp_ind]
            max_value = pred_ens_prior[0][comp_ind + start_index]
            min_value = pred_ens_prior[0][comp_ind + start_index]
            for ind_ne in range(ne):

                max_value = max(max_value, pred_ens_prior[ind_ne][comp_ind + start_index])
                min_value = min(min_value,
                                pred_ens_prior[ind_ne][comp_ind + start_index])

            noise_in_data_point_assessed = Cd_std[count_ind + dat_ind * num_components]
            max_value += noise_in_data_point_assessed * 2
            min_value -= noise_in_data_point_assessed * 2
            if max_value < data_point_assessed:  # cover_high[comp_ind]:

                coverage[count_ind + dat_ind * num_components] = True
            if min_value > data_point_assessed:  # cover_low[comp_ind]:

                coverage[count_ind + dat_ind * num_components] = True

        start_index += n_data_points
    if call_plot_coverage:
        plot_coverage(pred_ens_prior)
    #print(coverage)

    # Test for patterns across components for each data type
    # TODO tailor model refinement to type of data not covered
    #red_ext_inv_domain = False #updt_p_mean = False; #inc_p_spread = False
    for comp in range(num_components):
        pattern_data_types = sum(coverage[comp::num_components])
        if pattern_data_types > np.floor(len(coverage[comp::num_components]) / 2):
            #red_ext_inv_domain = True
            #updt_p_mean = True
            success_coverage = False
        print(f"Component {components[indices_to_keep[comp]]} pred. not covered by data in {pattern_data_types} tool settings")

    # Test for patterns across data types for each component
    for dt in range(num_data_types):
        pattern_components = sum(coverage[dt * num_components:(dt + 1) * num_components])
        if pattern_components > 2:#np.floor(len(coverage[dt * num_components:(dt + 1) * num_components]) / 2):
            #inc_p_spread = True
            success_coverage = False
        print(f"Tool setting {data_keys[dt]} pred. not covered by data in {pattern_components} components: {success_coverage}")


    return success_coverage#, inc_p_spread, updt_p_mean, red_ext_inv_domain

def modify_prior_to_increase_coverage(u_inf, incr_p_uncert, ens_m_kriging, comb_ga_kriging):
    #    def modify_prior_to_increase_coverage(increase_param_spread, coarsen_inversion_domain, u_f, input_dict,
    #                                          ext_x_update, ext_z_update, success, red_fact):
    # The spatial extent of the inversion domain is about 400 feets and hence contains >80 logging points
    # When the ensembel predictions do not cover the observations this is most likely due to an abrubt change in geology
    # ahead of the logging point
    # hence, continue using kriging for the first part and then impose a clean new prior for the tail of the inversion domain
    # Default:  Use kriging based on a weighted ensemble mean
        # Test 1. Use kriging based on individual ensemble members
        # Test 2. Add cells with large spread using smoothness information only
        # Test 3. Expanding ensemble spread by inflating parameter uncertainty
        # Final option is to run inversion with poor data coverage
    # Default settings:
    # ens_m_kriging = False: Test alt 1 first
    # comb_ga_kriging = False
    # incr_p_uncert = False
    tested_alt = False
    # Test 1. increase parameter uncertainty
    if u_inf < 0.05:
        if ens_m_kriging: # alt 1 has been tested, go to next alternative:
            if comb_ga_kriging: # alt 2 has been tested
                if incr_p_uncert: # alt 3 has been tested
                    tested_alt = True  # do inversion with current prior
                    print(f'Conduct inversion though prior predictions have limited data coverage')
                else:
                    u_inf += 0.1
                    print(f'Increase parameter uncertainty by {u_inf:0.2f} to {np.mean(pr[f"{param_keys[0]}_std"]) + u_inf:0.2f} and {np.mean(pr[f"{param_keys[1]}_std"]) + u_inf:0.2f} to secure data coverage')
            else: # alt 3
                comb_ga_kriging = True
                print(f'Add cells using prior mean {1} and  parameter uncertainty {0.25 * (1 + u_inf):0.2f}')
        else: # combine  alt 1 and alt 2
            ens_m_kriging = True
            comb_ga_kriging = True
            print(f'Make prior by kriging on individual ensemble members to secure data coverage; and')
            print(f'add cells using prior mean {1} and  parameter uncertainty {0.25 * (1 + u_inf):0.2f}')

    else: # alt 3 has been tested
        incr_p_uncert = True
        tested_alt = True  # do inversion with current prior
        print(f'Conduct inversion though prior predictions have limited data coverage')


    # TODO compare ordinary and universal kriging  - test different correlation lengts to improve data coverage
    # TODO maybe if only one has reached its limit, test the other before changing correlation lengths
    return tested_alt, u_inf, incr_p_uncert, ens_m_kriging, comb_ga_kriging #ext_x_update, ext_z_update, u_f, input_dict, red_fact_dx

def setup_obs_vect():
    Cd_row = Cd.iloc[assim_index[0]]
    # Cd_vec = np.concatenate([cell[1] for cell in Cd_row])
    data_vec = np.concatenate([data.iloc[assim_index[0]][dat][indices_to_keep] for dat in data_keys])
    data_vec_full = np.concatenate([data.iloc[assim_index[0]][dat] for dat in data_keys])
    max_per_key = {dat: data.iloc[assim_index[0]][dat].max() for dat in data_keys}
    val_list = [data.iloc[assim_index[0]][dat].max() for dat in data_keys]
    # add lower noise level on all data points of 1% of maximum data value
    Cd_std_added = np.concatenate(
        [np.sqrt(np.array(cell[1])) + 0.01 * val_list[i] for i, cell in enumerate(Cd_row)])
    Cd_std_full = Cd_std_added + 0.02 * abs(data_vec_full)

    Cd_std_added = np.concatenate(
        [np.sqrt(np.array(cell[1])[indices_to_keep]) + 0.01 * val_list[i] for i, cell in enumerate(Cd_row)])
    Cd_std = Cd_std_added + 0.02 * abs(data_vec)
    Cd_vec = np.square(Cd_std)
    data_real = np.random.normal(loc=data_vec, scale=Cd_std)
    return data_real, Cd_vec, Cd_std, Cd_std_full

param_vec_ens = []
tot_loss_ens =  []
failed_ensemble_member = np.zeros(ne, dtype=bool)

# inversion domain for this data point along the well trajectory
dX, dY, dZ = setup_inversion_grid(ext_x, ext_y, ext_z)
# for el,assim_index in enumerate(tot_assim_index[:15]):


UTA2D_input, TVD, MD, WX, WY = setup_input_dict(reference_model_path, data_type)
dtvd = vertical_spacing_log_points(10, TVD)
no_log_points_per_cell_z = _no_logging_points_per_grid_cell_z(dZ, dtvd)
posterior = []
for el, assim_index in enumerate(tot_assim_index[75:85:1]):

    # Extract the el-th row of Cd and concatenate variance values
    data_real, Cd_vec, Cd_std, Cd_std_full = setup_obs_vect()

    # Check prior model ensemble coverage before calibration with data
    success = False
    u_f = 0
    #ext_x_update = 0, ext_z_update = 0  #reduce_extent_inversion_domain = False
    # Default settings:
    if len(tot_loss_ens) == 0 and el == 0:
        first_assimilation_step = True
    else:
        first_assimilation_step = False

    use_ensemble_member_kriging = True # Test 1
    comb_gauss_kriging = False
    increase_param_uncertainty = False

    # check if the logging point reach a new grid cell;
    if el % no_log_points_per_cell_z == 0:
        no_logg_one_cell = 0
        if first_assimilation_step:
            shift_cells = True

        else:
            shift_cells = True
            print(f'shift inversion domain after {no_log_points_per_cell_z} logging points')
    else:
        no_logg_one_cell +=1
        shift_cells = True

    setup_simulators(UTA2D_input, assim_index[0]-no_logg_one_cell, TVD, MD, WX, WY, shift_cells)

    while not success:
        # prior ensemble of parameter values within inversion domain
        pr, Cm = gen_prior(posterior, dims[2], dims[0], dZ, dX,
                           ded, dtvd, u_f, comb_gauss_kriging, use_ensemble_member_kriging, first_assimilation_step, shift_cells)
        if first_assimilation_step:
            success = True
        else:
            success = True#check_coverage()
        # Default:  Use kriging based on a weighted ensemble mean
        # Test 1. Use kriging based on individual ensemble members
        # Test 2. Add new cells at the end  using smoothness information only
        # Test 3. Expanding ensemble spread by inflating parameter uncertainty
        # Final option is to run inversion with poor data coverage
        #success = False #(for debugging)
        if not success:
            tested_alternatives, u_f, increase_param_uncertainty, use_ensemble_member_kriging, comb_gauss_kriging = (
                modify_prior_to_increase_coverage(u_f, increase_param_uncertainty, use_ensemble_member_kriging, comb_gauss_kriging))
            if tested_alternatives:
                success = True
    tot_loss_ens = np.zeros([ne, 2])
    param_vec_ens = []

    for member in range(ne):
        # Initialize
        J = []
        J_full = []
        pred_vec = []
        pred_vec_full = []

        #if first_assimilation_step:
        param = {k: pr[k][:, member] for k in param_keys}
        param_vec = np.concatenate([param[key].flatten() for key in param_keys])
        #else:
        #    param_vec = posterior[member]

        tmp_param_vec = param_vec
        tot_loss = []
        debug = False
        if debug:
            plt.figure()
            plt.imshow(np.exp(param['rh'].reshape(nz, numb_horizontal_grid)), aspect='auto')
            plt.colorbar()
            plt.title(f'Iteration {0} member {member} - rh assim {assim_index[0]}')
            plt.savefig(f'rh_iter{0}_member{member}_assim{assim_index[0]}.png')


        # step_lambda = 10**np.floor(np.log10(tot_loss[-1]/2*len(pred_vec)))
        step_lambda = 1e5
        param_dict = {}
        for iter_ind in range(n_iter):
            offset = 0
            for key in param_keys:
                param_dict[key] = tmp_param_vec[offset:offset + param_size]
                offset += param_size #+ numb_horizontal_grid

            if isinstance(tmp_param_vec, (bool, np.bool_)):
                print('something went wrong')

            try:
                if UTA2D_input['jacobian']:
                    pred, jacobian = UTA2D_sim.run_fwd_sim(param_dict, 0)
                    J, J_full = construct_full_jacobian()
                else:
                    pred = UTA2D_sim.run_fwd_sim(param_dict, 0)
                    pred = UTA2D_sim.run_fwd_sim(param_dict, 0)
                    J = []
                    J_full = []
            except:
                # Move to next ensemble member
                failed_ensemble_member[member] = True
                print('ensemble failed')
                break


            pred_vec = np.concatenate([pred[assim_index[0]][k][indices_to_keep] for k in tools])

            pred_vec_full = np.concatenate([pred[assim_index[0]][k] for k in tools]) # for plotting
            # Construct full Jacobian matrix


            curr_loss, data_loss, param_loss = custom_loss(pred_vec, tmp_param_vec, pr)
            tot_loss.append(curr_loss)
            if iter_ind == 0:
                step_lambda = max(1e2,curr_loss/(2*len(pred_vec)))
                tot_loss_ens[member,iter_ind] = curr_loss
                # save prior param ensemble and data predictions to file
                with open(('ensemble_dump/prior_ensemble_member_' + str(member) + '_assim_' + str(assim_index[0]) + '.pkl'),
                            'wb') as f:
                        pickle.dump({'pred': pred_vec_full, 'misfit': tot_loss,
                                     'J': J_full, 'st_dev': Cd_std_full, 'st_dev_param': Cm,
                                     'param_vec': param_vec}, f)
                # save results based on prior ensemble mean
                with open(('ensemble_dump/prior_ensemble_mean_' + str(member) + '_assim_' + str(assim_index[0]) + '.pkl'),
                            'wb') as f:
                        pickle.dump({'pred': pred_vec_full, 'misfit': tot_loss,
                                     'J': J_full, 'st_dev': Cd_std_full, 'st_dev_param': Cm,
                                     'param_vec': param_vec}, f)

            print(f"Assimilation step {int(assim_index[0])}, Iteration {int(iter_ind)}, Loss: {int(curr_loss)}, Lambda {int(step_lambda)}, Ensemble member {member}")
            if iter_ind > 0:
                if tot_loss[-1] < tot_loss[-2]:
                    step_lambda /= 2
                    param_vec = tmp_param_vec
                else:
                    tot_loss = tot_loss[:-1]
                    step_lambda *= 2
                converged = has_converged(tot_loss, data_loss, param_loss, J, step_lambda)
                if converged:
                    tot_loss_ens[member, 1] = tot_loss[-1]
                    print(f"Convergence met")
                    break

            tmp_param_vec = map_update_data_space_LM(param_vec, pr, pred_vec, J, data_real, Cd_vec, step_lambda, Cm)
            if iter_ind == (n_iter - 1):
                tot_loss_ens[member, 1] = tot_loss[-1]

        param_vec_ens.append(param_vec)



        if debug:
            offset = 0
            for key in param_keys:
                plt.figure()
                if key == 'rh' or key == 'rv':
                    values = np.exp(param_vec[offset:offset + param_size].reshape(nz, numb_horizontal_grid))
                else:
                    values = param_vec[offset:offset + param_size].reshape(nz, numb_horizontal_grid)
                plt.imshow(values, aspect='auto')
                offset += param_size #+ numb_horizontal_grid
                plt.colorbar()
                plt.title(f'Member{member} param {key} assimilation step  {assim_index[0]}')
                plt.savefig(f'Member_{member}_param_{key}_assim_{assim_index[0]}.png')
                plt.close()


        with open(('ensemble_dump/ensemble_member_' + str(member) + '_assim_' + str(assim_index[0]) +'.pkl'), 'wb') as f:
            pickle.dump({'pred': pred_vec_full, 'misfit':tot_loss,
                     'J': J_full, 'st_dev': Cd_std_full, 'st_dev_param': Cm,
                     'param_vec': param_vec}, f)

    posterior = param_vec_ens
    # Calculate simulation results for the ensemble mean
    param_vec_ens_array = np.vstack(param_vec_ens)  # shape (n_ens, n_param)
    param_vec_ens_mean = np.mean(param_vec_ens_array, axis=0)
    offset = 0
    param_dict = {}
    for key in param_keys:
        param_dict[key] = param_vec_ens_mean[offset:offset + param_size]
        offset += param_size #+ numb_horizontal_grid
    if UTA2D_input['jacobian']:
        pred, jacobian = UTA2D_sim.run_fwd_sim(param_dict, 0)
        _, J_ens_mean = construct_full_jacobian()
        pred_vec_ens_mean = np.concatenate([pred[assim_index[0]][k] for k in tools])
        with open(('ensemble_dump/ensemble_mean' + '_assim_' + str(assim_index[0]) + '.pkl'), 'wb') as f:
            pickle.dump({'pred': pred_vec_ens_mean,
                         'J': J_ens_mean, 'st_dev': Cd_std_full, 'st_dev_param': Cm,
                         'param_vec': param_vec_ens_mean}, f)
    else:
        pred = UTA2D_sim.run_fwd_sim(param_dict, 0)
        J_ens_mean = []
        pred_vec_ens_mean = np.concatenate([pred[assim_index[0]][k] for k in tools])
        with open(('ensemble_dump/without_J_ensemble_mean' + '_assim_' + str(assim_index[0]) + '.pkl'), 'wb') as f:
            pickle.dump({'pred': pred_vec_ens_mean,
                         'J': J_ens_mean, 'st_dev': Cd_std_full, 'st_dev_param': Cm,
                         'param_vec': param_vec_ens_mean}, f)

    # Construct full Jacobian matrix



    with open(('ensemble_dump/case_setup_dict' + '_assim_' + str(assim_index[0]) +'.pkl'), 'wb') as f:
        pickle.dump({'input_dict': UTA2D_input}, f)