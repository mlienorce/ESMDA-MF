"""
Combine ESMDA with a gradient based optimizaiton method.
1. Do N_MDA - 1 ESMDA iterations
2. Do gradient based optimization for each ensemble member for the last MDA "iteration"
"""
from copy import deepcopy

import numpy as np
from EMsim.EM import UTA2D
from EMsim.EM import UTA0D
from EMsim.EM import UTA1D
import os
from scipy.stats import norm
import copy
import h5py
import argparse
from ThreeDGiGEarth.common import h5_to_dict
import pandas as pd
import ast
import pickle
import matplotlib.pyplot as plt
from pipt.update_schemes.update_methods_ns.approx_update import approx_update
from pipt.misc_tools.cov_regularization import localization
from geostat.decomp import Cholesky
from scipy.stats import truncnorm
from scipy.ndimage import gaussian_filter
import gc, os
import psutil

geostat = Cholesky()
## setup the model
for folder in os.listdir('.'):
    if folder.startswith('En_') and os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)

meter_to_feet = 3.28084

dims = (23,3,34) # number of cells in each direction (x,y,z)
#dims = (7,3,10) # number of cells in each direction (x,y,z)
cell_thickness_ft = 5  # horizontal spacing logging point
vertical_spacing_between_logging_points = 0.437#4433316006616
dX = cell_thickness_ft * 1
dY = cell_thickness_ft * 1 * 4
dZ = vertical_spacing_between_logging_points * (12)

# option to keep cells fixed far from bit:
no_fixed_cells_z = 4 # if set to zero does not apply


#help(geostat.gen_cov2d)
var_range_m = 7.0
var_range_ft = var_range_m * meter_to_feet
var_range_cells = max(1, int(np.floor(var_range_ft / dX)))
vertical_aspect = (4 * dZ) / dX

#reference_model_path = '/home/AD.NORCERESEARCH.NO/krfo/CodeProjects/DISTINGUISH/Jacobian/inversion/data/Benchmark-3/globalmodel.h5'
reference_model_path = '/home/AD.NORCERESEARCH.NO/mlie/3DGiG/Jacobian/inversion/data/Benchmark-3'
reference_model = '/home/AD.NORCERESEARCH.NO/mlie/3DGiG/Jacobian/inversion/data/Benchmark-3/globalmodel.h5'
Ne_MDA = 250 # ES-MDA
Ne_opt = 10 # gradient based
Ne = Ne_MDA

debug = True

# Boundaries for rh: # These are physical values-not log-values
rh_phys_min = 0.5
rh_phys_max = 70
rv_phys_max = 90
apply_bounds = True
# observed_data_order_udar = [
#         'USDP', 'USDA',
#         'UADP', 'UADA',
#         'UHRP', 'UHRA',
#         'UHAP', 'UHAA'
#     ]
# selected_data = ['USDP', 'USDA', 'UADP', 'UADA','UHRP', 'UHRA','UHAP', 'UHAA']
# #selected_data = ['USDA', 'UADA'] 
# selected_data_indices = [observed_data_order_udar.index(data) for data in selected_data]

observed_data_order_bfield = ['real(Bxx)', 'real(Bxy)', 'real(Bxz)',
                            'real(Byx)', 'real(Byy)', 'real(Byz)',
                            'real(Bzx)', 'real(Bzy)', 'real(Bzz)',
                            'img(Bxx)', 'img(Bxy)', 'img(Bxz)',
                            'img(Byx)', 'img(Byy)', 'img(Byz)',
                            'img(Bzx)', 'img(Bzy)', 'img(Bzz)']

observed_data_order_0D_bfield = ['real(Bxx)','img(Bxx)',
                                'real(Byy)','img(Byy)',
                                'real(Bzz)','img(Bzz)',
                                'real(Bxz)','img(Bxz)',
                                'real(Bzx)','img(Bzx)']

selected_data = ['real(Bxx)',
                # 'real(Bxy)',
                 'real(Bxz)',
                #'real(Byx)', 'real(Byy)', 'real(Byz)',
                'real(Bzx)', 
                #'real(Bzy)',
                'real(Bzz)',
                'img(Bxx)',
                #'img(Bxy)',
                'img(Bxz)',
                #'img(Byx)', 'img(Byy)', 'img(Byz)',
                'img(Bzx)',
                #'img(Bzy)',
                'img(Bzz)']
selected_data_indices = [observed_data_order_bfield.index(data) for data in selected_data]
selected_data_indices_0D = [observed_data_order_0D_bfield.index(data) for data in selected_data if data in observed_data_order_0D_bfield]
selected_data_indices_1D = selected_data_indices_0D

min_ratio, max_ratio = 1, 4 # Bounds for the anisotropy ratio

tools = ["('6kHz','83ft')","('12kHz','83ft')","('24kHz','83ft')",
         "('24kHz','43ft')","('48kHz','43ft')","('96kHz','43ft')"]
#tools = ["('24kHz','83ft')"] #, "('96kHz','43ft')"]
#tools = ["('6kHz','83ft')","('12kHz','83ft')",
#         "('24kHz','43ft')","('48kHz','43ft')"]
#tools = ["('12kHz','83ft')","('24kHz','83ft')",
#        "('96kHz','43ft')"]
#tools = ["('96kHz','43ft')"] #

def setup_simulators(ref_mod_path, data_type):
    """
    Set up the UTA0D, UTA1D, and UTA2D simulators with the given reference model path and data type.
    """


    UTA_input_dict = {'toolflag': 0,
                'datasign': 3 if data_type == 'UDAR' else 0,
                'anisoflag': 1,
                'toolsetting': f'{ref_mod_path}/ascii/tool.inp',
                'trajectory': f'{ref_mod_path}/ascii/trajectory.DAT',
                'reference_model': f'{ref_mod_path}/globalmodel.h5',
                'parallel': 1,
                'map':{'ratio': [min_ratio,max_ratio]},
            }

    with h5py.File(UTA_input_dict['reference_model'], "r") as f:
        ref_model = h5_to_dict(f)

    wellpath = ref_model['wellpath']
    TVD = wellpath['Z'][:,0] * meter_to_feet  # in feet
    MD = wellpath['Distance'][:,0] * meter_to_feet  # in feet
    WX = wellpath['X'][:,0] * meter_to_feet  # in feet
    WY = wellpath['Y'][:,0] * meter_to_feet  # in feet

    sim_info = {
    'obsname': 'tvd',
    'assimindex': [[0]],
    'datatype': tools
    }

    # Setup UTA0D simulator
    UTA0D_input = {key: UTA_input_dict[key] for key in ['toolflag', 'datasign', 'anisoflag', 'reference_model', 'map',
                                                        'toolsetting']}
    UTA0D_sim = UTA0D({**UTA0D_input, **sim_info})
    UTA0D_sim.setup_fwd_run(redund_sim=None)


    # Setup UTA1D simulator
    initial_surface_depth = [TVD[0] - 0.5 * dims[2] * dZ + i * dZ for i in range(dims[2])]
    UTA1D_input = {key: UTA_input_dict[key] for key in ['toolflag', 'datasign', 'anisoflag', 'reference_model', 'map',
                                                        'toolsetting', 'trajectory']}
    UTA1D_input['surface_depth'] = initial_surface_depth
    UTA1D_sim = UTA1D({**UTA1D_input, **sim_info})
    UTA1D_sim.setup_fwd_run(redund_sim=None)

    # Setup UTA2D simulator
    UTA2D_input = {key: UTA_input_dict[key] for key in ['toolflag', 'datasign', 'anisoflag','reference_model','map',
                                                        'toolsetting']}
    UTA2D_input['dims'] = dims
    UTA2D_input['dX'] = dX
    UTA2D_input['dY'] = dY
    UTA2D_input['dZ'] = dZ
    UTA2D_input['shift'] = {'x':0,
                            'y':0,
                            'z':0}
    UTA2D_input['jacobian'] = True

    UTA2D_sim = UTA2D({**UTA2D_input, **sim_info},{'jacobi': True})
    UTA2D_sim.setup_fwd_run(redund_sim=None)

    with open(('Results/case_setup_dict' +'.pkl'), 'wb') as f:
        pickle.dump({'input_dict': UTA2D_input}, f)

    return UTA0D_sim, UTA1D_sim, UTA2D_sim, TVD, MD, WX, WY, wellpath

def sample_prior_0D(ne, normal_rh_mean_0d, normal_rh_std_0d):
    #sample the 0D prior, that is a univariate Gaussian for log-resistivity and a standard normal for the anisotropy ratio latent variable, without correlation
    # Create 2x2 diagonal covariance matrix for [log_rh, ratio_latent]
    c_theta_0d = np.diag([normal_rh_std_0d**2, ratio_latent_std**2])
    normal_rh_samples = np.random.normal(loc=normal_rh_mean_0d, scale=normal_rh_std, size=(ne,))
    ratio_latent_samples = np.random.normal(loc=ratio_latent_mean, scale=ratio_latent_std, size=(ne,))
    
    # Stack samples: shape (2, ne)
    samples = np.vstack([normal_rh_samples, ratio_latent_samples])
    
    mean_theta_0d = np.array([normal_rh_mean, ratio_latent_mean])

    return samples, c_theta_0d, mean_theta_0d

def normal_std_from_lognormal_std(sigma_X, mu):
    """
    Calculate the standard deviation of the normal distribution
    associated with a given log-normal standard deviation.

    Parameters:
    - sigma_X: Standard deviation of the log-normal distribution
    - mu: Mean of the normal distribution

    Returns:
    - sigma: Standard deviation of the normal distribution
    """

    # Initial guess for sigma
    sigma_guess = 1.0  # Starting guess

    # Function to compute the difference we want to be 0
    def equation(sigma):
        return np.sqrt((np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)) - sigma_X

    # Find the standard deviation of the normal distribution
    from scipy.optimize import fsolve
    sigma_solution = fsolve(equation, sigma_guess)[0]

    return sigma_solution

def normal_bounded_std_from_lognormal_std(sigma_X, mu):
    """
    Calculate the standard deviation of the normal distribution
    associated with a given log-normal standard deviation.

    Parameters:
    - sigma_X: Standard deviation of the log-normal distribution
    - mu: Mean of the normal distribution for the bounded parameter

    Returns:
    - sigma: Standard deviation of the normal distribution for the bounded parameter
    """

    a = rh_phys_min
    b = rh_phys_max

    # invert m_b mean to representative rh0: m = exp(mu_mb)
    m = np.exp(mu)
    rh0 = (m * b + a) / (1.0 + m)  # from rh = (m*b + a)/(1+m)
    # derivative dm_b/drh = (b - a) / ((rh - a)*(b - rh))
    deriv = (b - a) / ((rh0 - a) * (b - rh0))
    sigma_solution =  np.abs(deriv) * sigma_X


    return sigma_solution

def find_std_keeping_variance_fixed(new_rh_mean, normal_rh_mean_target, normal_rh_std_target):
    # Known values

    # Calculate target variance for log_normal distribution
    target_variance = (np.exp(normal_rh_std_target ** 2) - 1) * np.exp(2 * normal_rh_mean_target + normal_rh_std_target ** 2)

    def equation_to_solve(log_rh_std):
        return (np.exp(log_rh_std ** 2) - 1) * np.exp(2 * new_rh_mean + log_rh_std ** 2) - target_variance

    log_normal_rh_std_initial_guess = 1.0
    log_normal_rh_std_solution = np.nan

    try:
        from scipy.optimize import fsolve
        log_normal_rh_std_solution = fsolve(equation_to_solve, log_normal_rh_std_initial_guess)[0]
    except Exception as e:
        print(f"Could not find solution: {e}")

    return log_normal_rh_std_solution

def compute_log_normal_params(mu, sigma):
    """
    Compute log-normal distribution parameters based on normal distribution parameters.

    Parameters:
    - mu: Mean of the normal distribution
    - sigma: Standard deviation of the normal distribution

    Returns:
    - mean_X, std_dev_X: Mean and standard deviation of the log-normal distribution
    """
    # Mean of the log-normal distribution
    mean_X = np.exp(mu + (sigma ** 2) / 2)

    # Variance of the log-normal distribution
    variance_X = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)

    # Standard deviation of the log-normal distribution
    std_dev_X = np.sqrt(variance_X)

    return mean_X, std_dev_X

def resample_prior_0D_to_1D(ne, param_vec_0d):
    """Resample a 0D ensemble into vertically correlated 1D columns."""


    param_vec_0d = np.asarray(param_vec_0d, dtype=float)
    if param_vec_0d.shape != (2, ne):
        raise ValueError(
            f"Expected param_vec_0d with shape (2, {ne}), got {param_vec_0d.shape}"
        )


    nz = dims[2]
    center_idx = nz // 2

    column_height_ft = nz * dZ
    var_range_m = column_height_ft / meter_to_feet
    var_range_ft = var_range_m * meter_to_feet
    var_range_cells = max(1, int(np.ceil(var_range_ft / dZ)))

    param_vec_1d = np.empty((2 * nz, ne), dtype=float)
    cov_blocks = []
    mean_blocks = []

    # set up rh mean and std used for making covariance
    normal_rh_mean_0d = float(np.median(param_vec_0d[0]))
    if apply_bounds:
        normal_rh_std_0d = normal_bounded_std_from_lognormal_std(log_normal_rh_std_target, normal_rh_mean_0d)
    else:
        normal_rh_std_0d = normal_std_from_lognormal_std(log_normal_rh_std_target, normal_rh_mean_0d)
    oneD_std = [normal_rh_std_0d, ratio_latent_std]  # Standard deviations for the 1D fields, can be tuned

    for param_idx in range(2):
        stationary_mean = float(np.mean(param_vec_0d[param_idx]))
        stationary_std = float(oneD_std[param_idx])
        mean_vec = np.full(nz, stationary_mean, dtype=float)
        mean_blocks.append(mean_vec)

        if stationary_std <= np.finfo(float).eps:
            cov = np.eye(nz, dtype=float) * np.finfo(float).eps
            conditioned_samples = np.tile(mean_vec[:, np.newaxis], (1, ne))
            conditioned_samples[center_idx, :] = param_vec_0d[param_idx, :]
        else:
            cov = geostat.gen_cov2d(
                nz, 1, stationary_std**2, var_range_cells, 1, 0, 'exp'
            )
            unconditional_samples = geostat.gen_real(mean_vec, cov, ne).reshape(nz, ne, order='C')
            kriging_weights = cov[:, center_idx] / cov[center_idx, center_idx]
            conditioning_residual = param_vec_0d[param_idx, :] - unconditional_samples[center_idx, :]
            conditioned_samples = unconditional_samples + kriging_weights[:, np.newaxis] * conditioning_residual[np.newaxis, :]
            conditioned_samples[center_idx, :] = param_vec_0d[param_idx, :]

            if apply_bounds:
                if param_idx == 0:
                    conditioned_samples = clip_minmax(conditioned_samples, latent_rh_min, latent_rh_max)
                else:
                    conditioned_samples = clip_minmax(conditioned_samples, latent_ratio_min, latent_ratio_max)


        start = param_idx * nz
        stop = start + nz
        param_vec_1d[start:stop, :] = conditioned_samples
        cov_blocks.append(cov)

    c_theta_1d = np.block([
        [cov_blocks[0], np.zeros_like(cov_blocks[0])],
        [np.zeros_like(cov_blocks[1]), cov_blocks[1]],
    ])
    mean_theta_1d = np.concatenate(mean_blocks)

    return param_vec_1d, c_theta_1d, mean_theta_1d

def resample_prior_multiple_1D_to_2D(ne, param_vec_1d_list, param_1d_columns):
    """Resample a 1D ensemble into 2D fields conditioned on the central column."""

    #param_vec_1D = np.asarray(param_vec_1D, dtype=float)
    nz = dims[2]
    nx = dims[0]
    n_params_per_type = nx * nz

    param_idx = 0
    mean_vec_1d = []
    for column_idx_no, column_idx in enumerate(param_1d_columns):
        param_vec_1d = param_vec_1d_list[column_idx_no]
        mean_profile = np.mean(param_vec_1d[param_idx * nz:(param_idx + 1) * nz, :], axis=1)
        mean_vec_1d.append(mean_profile)
    if apply_bounds:
        normal_rh_mean_1d = np.median(mean_vec_1d)
        normal_rh_std_1d = normal_bounded_std_from_lognormal_std(log_normal_rh_std_target, normal_rh_mean_1d)
    else:
        normal_rh_mean_1d = min(1.5, np.max(np.median(mean_vec_1d)))
        normal_rh_std_1d = normal_std_from_lognormal_std(log_normal_rh_std_target, normal_rh_mean_1d)

    twoD_std = [normal_rh_std_1d, ratio_latent_std]  # Standard deviations for the 1D fields, can be tuned

    param_vec_2d = np.empty((2 * n_params_per_type, ne), dtype=float)
    cov_blocks = []
    mean_blocks = []

    for param_idx in range(2):
        mean_vec = np.zeros(nz*nx)
        memberwise_trend = np.zeros([nz*nx,ne])
        all_conditioning_indices = []
        all_conditioning_columns = []
        for column_idx_no, column_idx in enumerate(param_1d_columns):
            param_vec_1d = param_vec_1d_list[column_idx_no]
            conditioning_indices = column_idx * nz + np.arange(nz)
            if param_vec_1d.shape != (2 * nz, ne):
                raise ValueError(
                    f"Expected param_vec_1D with shape ({2 * nz}, {ne}), got {param_vec_1d.shape}"
                )
            conditioning_column = param_vec_1d[param_idx * nz:(param_idx + 1) * nz, :]
            mean_profile = np.mean(conditioning_column, axis=1)
            if column_idx_no == 0:
                mean_vec = np.tile(mean_profile, nx)
                memberwise_trend = np.tile(conditioning_column, (nx, 1))
            mean_vec[conditioning_indices] = mean_profile
            memberwise_trend[conditioning_indices, :] = conditioning_column
            if column_idx_no == len(param_1D_columns):
                mean_vec[column_idx * nz : nx * nz] = np.tile(mean_profile, nx-column_idx)
                memberwise_trend[column_idx * nz : nx * nz, :] = np.tile(conditioning_column, (nx-column_idx, 1))
            all_conditioning_indices.append(conditioning_indices)
            all_conditioning_columns.append(conditioning_column)
        all_conditioning_indices = np.concatenate(all_conditioning_indices, axis=0)
        all_conditioning_columns = np.concatenate(all_conditioning_columns, axis=0)

        stationary_std = float(twoD_std[param_idx])
        # Keep the prior mean as the ensemble-average 1D trend, but generate
        # each 2D realization around its own 1D column so the full center
        # column imprint propagates more uniformly across x.
        #mean_vec = np.tile(mean_profile, nx)
        mean_blocks.append(mean_vec)
        #memberwise_trend = np.tile(conditioning_column, (nx, 1))

        if stationary_std <= np.finfo(float).eps:
            cov = np.eye(n_params_per_type, dtype=float) * np.finfo(float).eps
            conditioned_residuals = np.zeros((n_params_per_type, ne), dtype=float)
        else:
            cov = geostat.gen_cov2d(
                nx, nz, stationary_std**2, var_range_cells, vertical_aspect, 0, 'exp'
            )
            unconditional_residuals = geostat.gen_real(
                np.zeros(n_params_per_type, dtype=float), cov, ne
            ).reshape(
                n_params_per_type, ne, order='C'
            )

            C_dd = cov[np.ix_(all_conditioning_indices, all_conditioning_indices)]
            C_dd = C_dd + np.eye(C_dd.shape[0], dtype=float) * np.finfo(float).eps
            C_xd = cov[:, all_conditioning_indices]
            conditioning_residual = -unconditional_residuals[all_conditioning_indices, :]

            try:
                kriging_weights = np.linalg.solve(C_dd, conditioning_residual)
            except np.linalg.LinAlgError:
                kriging_weights = np.linalg.pinv(C_dd) @ conditioning_residual

            conditioned_residuals = unconditional_residuals + C_xd @ kriging_weights
            conditioned_residuals[all_conditioning_indices, :] = 0.0

        conditioned_samples = memberwise_trend + conditioned_residuals
        conditioned_samples[all_conditioning_indices, :] = all_conditioning_columns

        start = param_idx * n_params_per_type
        stop = start + n_params_per_type
        param_vec_2d[start:stop, :] = conditioned_samples
        cov_blocks.append(cov)

    c_theta_2d = np.block([[cov_blocks[0], np.zeros_like(cov_blocks[0])],
        [np.zeros_like(cov_blocks[1]), cov_blocks[1]],])
    mean_theta_2d = np.concatenate(mean_blocks)

    return param_vec_2d, c_theta_2d, mean_theta_2d

def adjust_param_for_rv_bounds(param, rh_unbounded, rv_min, rv_max, eps=1e-12):
    """
    param, rh_unbounded: arrays (same shape) or scalars
    returns: param_adj (same shape), n_changed
    """
    import numpy as np
    try:
        from scipy.stats import norm
        _ppf = norm.ppf
        _cdf = norm.cdf
    except Exception:
        from math import sqrt
        from scipy.special import erfinv
        def _ppf(u):
            return np.sqrt(2) * erfinv(2 * u - 1)

        def _cdf(x):
            return 0.5 * (1 + np.erf(x / np.sqrt(2)))

    param = np.asarray(param)
    rh_unbounded = np.asarray(rh_unbounded)
    assert param.shape == rh_unbounded.shape

    R = float(max_ratio) / float(min_ratio)
    log_min = np.log(min_ratio)
    if np.isclose(R, 1.0):
        # ratio fixed; only check if rv within bounds; no degree of freedom
        rv = rh_unbounded + log_min
        mask_low = rv < rv_min
        mask_high = rv > rv_max
        mask = mask_low | mask_high
        # cannot change param to fix (or change to arbitrary value)
        return param.copy(), int(np.count_nonzero(mask))

    logR = np.log(R)

    u = _cdf(param)                       # in (0,1)
    # compute current rv
    log_ratio = log_min + u * logR
    rv = rh_unbounded + log_ratio

    # mask where rv violates bounds
    mask_low  = rv < rv_min
    mask_high = rv > rv_max
    mask = mask_low | mask_high
    if not np.any(mask):
        return param.copy()

    # compute allowed u interval solving:
    # rv_min <= rh + log_min + u*logR <= rv_max
    # => u_low = (rv_min - rh - log_min) / logR
    #    u_high = (rv_max - rh - log_min) / logR
    u_low  = (rv_min - rh_unbounded - log_min) / logR
    u_high = (rv_max - rh_unbounded - log_min) / logR

    # clamp to [0,1]
    u_low_clamped  = np.clip(u_low,  0.0, 1.0)
    u_high_clamped = np.clip(u_high, 0.0, 1.0)

    # for each violating element, choose the u in [u_low_clamped,u_high_clamped]
    # that is closest to the original u (minimize change)
    u_adj = u.copy()
    # elements where allowed interval is empty (u_low_clamped > u_high_clamped) are handled by clipping
    # compute midpoint candidate and also clip original u into interval
    u_clipped = np.minimum(np.maximum(u, u_low_clamped), u_high_clamped)
    # if you prefer minimal L2 change, u_clipped is the nearest point in interval to u
    u_adj[mask] = u_clipped[mask]

    # avoid exact 0/1 for ppf
    u_adj = np.clip(u_adj, eps, 1.0 - eps)
    param_adj = _ppf(u_adj)

    # return adjusted param (only changed where needed)
    #n_changed = int(np.count_nonzero(mask & (np.abs(param_adj - param) > 0)))
    return param_adj

def resample_prior_1D_to_2D_simplified(param_vec_1d_list, param_1d_columns):
    """Resample a 1D ensemble into 2D fields conditioned on the central column."""

    # param_vec_1d_list: nx // 2 + 1 columns. The first nx // 2 columns are the Ne_opt 2d results from previous assimilation.
    # The last column is the Ne_opt best 1d result from this assimilation
    nz = dims[2]
    nx = dims[0]
    n_params_per_type = nx * nz

    param_vec_2d = np.empty((2 * n_params_per_type, Ne_opt), dtype=float)
    cov_blocks = []
    mean_blocks = []

    # Standard deviations for the 1D fields, can be tuned
    param_idx = 0
    mean_vec_2d = []
    for column_idx_no, column_idx in enumerate(param_1d_columns):
        param_vec_1d = param_vec_1d_list[column_idx_no]
        mean_profile = np.mean(param_vec_1d[param_idx * nz:(param_idx + 1) * nz, :], axis=1)
        mean_vec_2d.append(mean_profile)
    if apply_bounds:
        normal_rh_mean_1d = np.median(mean_vec_2d)
        normal_rh_std_2d = normal_bounded_std_from_lognormal_std(log_normal_rh_std_target, normal_rh_mean_1d)
    else:
        normal_rh_mean_1d = min(1.5, np.max(np.median(mean_vec_2d)))
        normal_rh_std_2d = normal_std_from_lognormal_std(log_normal_rh_std_target, normal_rh_mean_1d)

    twoD_std = [normal_rh_std, ratio_latent_std]

    # ************ Build prior ensemble and covariance matrix for 2d inversion *************************
    #
    best_idx = np.argsort(post_loss)[:Ne_opt // 2]
    worst_idx = np.argsort(post_loss)[Ne_opt // 2:]

    for param_idx in range(2):
        #
        # Alternative 1: 2d ensemble from adjusting previous 2d ensemble relative to new logging position
        #
        reshaped_idx = []#np.array([])
        for en_count,post_param_one_ens_member in enumerate(post_param):  # initialize with previous 2D results
            # implement this shift of the 2D estimate at previous logging point only if inversion grid is moved
            reshaped_org = post_param_one_ens_member.reshape(2, dims[0] * dims[2])
            if not keep_inversion_lim_x:
                # Alt 1. duplicate last column
                reshaped = np.concatenate((reshaped_org[:, nz:], reshaped_org[:, -nz:]), axis=1)
                if en_count > np.floor(Ne_opt // 2):
                    # Alt 2. make a new guess
                    if apply_bounds:
                        prior_0d, _, _ = sample_prior_0D(Ne_MDA // 2, normal_rh_bounded_mean, normal_rh_bounded_std)
                    else:
                        prior_0d, _, _ = sample_prior_0D(Ne_MDA // 2, normal_rh_mean, normal_rh_std)

                    _, _, mean_prior_1d = resample_prior_0D_to_1D(Ne_MDA // 2, prior_0d)
                    reshaped[:, -nz:] = mean_prior_1d.reshape(2, nz)

            else:
                reshaped = reshaped_org

            if not keep_inversion_lim_z:
                arr = reshaped.reshape(2, nx, nz)
                # build new array where for each indx we drop indz=0 and append a copy of indz=nz-1
                # result has same shape (2, nx, nz)
                new_arr = np.empty_like(arr)
                # take original indz 1..nz-1
                new_arr[:, :, : nz - 1] = arr[:, :, 1: nz]
                # Alt 1. duplicate original indz = nz-1 into last position
                new_arr[:, :, nz - 1] = arr[:, :, nz - 1]

                # flatten back to original 2D layout if needed
                reshaped = new_arr.reshape(2, nx * nz)

            reshaped_idx.append(reshaped[param_idx, :])

        param_2d_alt1 =  np.array(reshaped_idx).T
        if apply_bounds:
            if param_idx == 0:
                param_2d_alt1 = clip_minmax(param_2d_alt1, latent_rh_min, latent_rh_max)
            else:
                param_2d_alt1 = clip_minmax(param_2d_alt1, latent_ratio_min, latent_ratio_max)


        # Alternative 2: 2d ensemble after conditioning to latest 1d result

        param_vec_1d  = param_vec_1d_list[-1][param_idx*nz:(param_idx+1)*nz] # 1d estimate from this assimilation step
        #
        mean_vec = (param_vec_1d.sum(axis=1) + mean_profile) / (param_vec_1d.shape[1] + 1)  # shape (15,)

        stationary_std = float(twoD_std[param_idx])

        cov = geostat.gen_cov2d(nx, nz, stationary_std**2, var_range_cells, vertical_aspect, 0, 'exp')
        mean_vec_tiled = np.tile(mean_vec, nx)
        if apply_bounds:
            param_2d_alt2 = apply_tighter_bounds_for_prior_ensemble(n_params_per_type,Ne_opt, param_idx, cov, mean_vec_tiled)
        else:
            param_2d_alt2 = geostat.gen_real(np.tile(mean_vec,nx), cov, Ne_opt).reshape(n_params_per_type, Ne_opt, order='C')


        # Combine the results from the two alternatives
        start = param_idx * n_params_per_type
        stop = start + n_params_per_type

        stop_alt1 = start + nz*(nx // 2)
        start_alt2 = stop_alt1
        stop_alt2 = stop - nz
        param_vec_2d[start:stop, best_idx] = param_2d_alt1[:,best_idx] # result from previous 2D inversion
        param_vec_2d[start:stop, worst_idx] = param_2d_alt1[:, best_idx]  # result from previous 2D inversion
        param_vec_2d[start_alt2:stop_alt2, worst_idx] = param_2d_alt2[(start_alt2-start):(stop_alt2-start), best_idx] # 1D result after logging point
        param_vec_2d[start_alt2:stop_alt2, worst_idx[0]] = mean_vec_tiled[(start_alt2 - start):(stop_alt2 - start)]  # 1D result after logging point

        if apply_bounds:
            if param_idx == 0:
                param_vec_2d[start:stop, :] = clip_minmax(param_vec_2d[start:stop, :], latent_rh_min, latent_rh_max)
            else:
                param_vec_2d[start:stop, :] = clip_minmax(param_vec_2d[start:stop, :], latent_ratio_min,
                                                          latent_ratio_max)
                if apply_bounds:  # param is m_bounded
                    rh_2d_unbounded = remove_bounds_rh(param_vec_2d[:n_params_per_type, :])
                else:
                    rh_2d_unbounded = param_vec_2d[:n_params_per_type, :]

                param_vec_2d[start:stop, :] = adjust_param_for_rv_bounds(param_vec_2d[start:stop, :], rh_2d_unbounded, rh_phys_min, rv_phys_max, eps=1e-12)

        mean_blocks.append(np.mean(param_vec_2d[start:stop],axis=1))
        cov_blocks.append(cov)

    c_theta_2d = np.block([[cov_blocks[0], np.zeros_like(cov_blocks[0])],
        [np.zeros_like(cov_blocks[1]), cov_blocks[1]],])
    mean_theta_2d = np.concatenate(mean_blocks)

    return param_vec_2d, c_theta_2d, mean_theta_2d

def vertical_spacing_log_points(logging_point_idx):
    dtvd = TVD[logging_point_idx]-TVD[logging_point_idx-assim_step_integer]
    return dtvd

def horizontal_spacing_log_points(logging_point_idx):
    dwx = WX[logging_point_idx] -WX[logging_point_idx-assim_step_integer]
    return dwx

def _no_logging_points_per_grid_cell(d_cell, d_logg):
    no_log_points_per_cell = np.ceil(d_cell / d_logg)
    return no_log_points_per_cell

def logging_point_vs_inversion_domain():

    if el == 0:
        keep_inversion_lim_z = False
        keep_inversion_lim_x = False
    else:
        # check if the logging point reach a new grid cell;
        dtvd = vertical_spacing_log_points(assim_index[0])
        no_log_points_per_cell_z = _no_logging_points_per_grid_cell(dZ, dtvd)
        dwx = horizontal_spacing_log_points(assim_index[0])
        no_log_points_per_cell_x = _no_logging_points_per_grid_cell(dX, dwx)
        if el <= no_log_points_per_cell_z // 2:
            keep_inversion_lim_z = True
        elif (el + no_log_points_per_cell_z // 2) % no_log_points_per_cell_z == 0:
            keep_inversion_lim_z = False
            print(f'shift inversion domain in z after {no_log_points_per_cell_z} logging points')
        else:
            keep_inversion_lim_z = True

        if el <= no_log_points_per_cell_x // 2:
            keep_inversion_lim_x = True
        elif (el + no_log_points_per_cell_x // 2) % no_log_points_per_cell_x == 0:
            keep_inversion_lim_x = False
            print(f'shift inversion domain in x after {no_log_points_per_cell_x} logging points')
        else:
            keep_inversion_lim_x = True

    return keep_inversion_lim_z, keep_inversion_lim_x

def resample_prior_1D_to_2D(ne, param_vec_1D):
    """Resample a 1D ensemble into 2D fields conditioned on the central column."""

    param_vec_1D = np.asarray(param_vec_1D, dtype=float)
    nz = dims[2]
    nx = dims[0]
    n_params_per_type = nx * nz

    if param_vec_1D.shape != (2 * nz, ne):
        raise ValueError(
            f"Expected param_vec_1D with shape ({2 * nz}, {ne}), got {param_vec_1D.shape}"
        )


    if apply_bounds:
        normal_rh_std_1d = normal_bounded_std_from_lognormal_std(log_normal_rh_std_target, np.median(param_vec_1D))
    else:
        normal_rh_std_1d = normal_std_from_lognormal_std(log_normal_rh_std_target, normal_rh_mean)

    twoD_std = [normal_rh_std_1d, ratio_latent_std]  # Standard deviations for the 1D fields, can be tuned
    center_x_idx = nx // 2
    conditioning_indices = center_x_idx * nz + np.arange(nz)

    param_vec_2D = np.empty((2 * n_params_per_type, ne), dtype=float)
    cov_blocks = []
    mean_blocks = []

    for param_idx in range(2):
        conditioning_column = param_vec_1D[param_idx * nz:(param_idx + 1) * nz, :]
        mean_profile = np.mean(conditioning_column, axis=1)
        stationary_std = float(twoD_std[param_idx])
        # Keep the prior mean as the ensemble-average 1D trend, but generate
        # each 2D realization around its own 1D column so the full center
        # column imprint propagates more uniformly across x.
        mean_vec = np.tile(mean_profile, nx)
        mean_blocks.append(mean_vec)
        memberwise_trend = np.tile(conditioning_column, (nx, 1))

        if stationary_std <= np.finfo(float).eps:
            cov = np.eye(n_params_per_type, dtype=float) * np.finfo(float).eps
            conditioned_residuals = np.zeros((n_params_per_type, ne), dtype=float)
        else:
            cov = geostat.gen_cov2d(
                nx, nz, stationary_std**2, var_range_cells, vertical_aspect, 0, 'exp'
            )
            unconditional_residuals = geostat.gen_real(
                np.zeros(n_params_per_type, dtype=float), cov, ne
            ).reshape(
                n_params_per_type, ne, order='C'
            )

            C_dd = cov[np.ix_(conditioning_indices, conditioning_indices)]
            C_dd = C_dd + np.eye(C_dd.shape[0], dtype=float) * np.finfo(float).eps
            C_xd = cov[:, conditioning_indices]
            conditioning_residual = -unconditional_residuals[conditioning_indices, :]

            try:
                kriging_weights = np.linalg.solve(C_dd, conditioning_residual)
            except np.linalg.LinAlgError:
                kriging_weights = np.linalg.pinv(C_dd) @ conditioning_residual

            conditioned_residuals = unconditional_residuals + C_xd @ kriging_weights
            conditioned_residuals[conditioning_indices, :] = 0.0

        conditioned_samples = memberwise_trend + conditioned_residuals
        conditioned_samples[conditioning_indices, :] = conditioning_column

        if apply_bounds:
            if param_idx == 0:
                conditioned_samples = clip_minmax(conditioned_samples, latent_rh_min, latent_rh_max)
            else:
                conditioned_samples = clip_minmax(conditioned_samples, latent_ratio_min, latent_ratio_max)


        start = param_idx * n_params_per_type
        stop = start + n_params_per_type
        param_vec_2D[start:stop, :] = conditioned_samples
        cov_blocks.append(cov)

    C_theta_2D = np.block([[cov_blocks[0], np.zeros_like(cov_blocks[0])],
        [np.zeros_like(cov_blocks[1]), cov_blocks[1]],])
    mean_theta_2D = np.concatenate(mean_blocks)

    return param_vec_2D, C_theta_2D, mean_theta_2D

def resample_post_2D_to_0D(post_param):

    center_idx = dims[0] // 2
    #center_idz = dims[2] // 2
    c_param_1 = []
    c_param_2 = []
    for post_param_one_ens_member in post_param:
        reshaped = post_param_one_ens_member.reshape(2, dims[0]*dims[2])
        c_param_1_value = np.mean(reshaped[0, :])
        c_param_2_value = np.mean(reshaped[1, :])
        c_param_1.append(c_param_1_value)
        c_param_2.append(c_param_2_value)

    c_param_1 = np.array(c_param_1)
    c_param_2 = np.array(c_param_2)
    # alt 1 use results from 2d inversion in logging point
    rh_mean = np.mean(c_param_1)
    ratio_latent_mean = np.mean(c_param_2)
    # alt 2 reset value to initial value based on geology only



    return rh_mean, ratio_latent_mean

def resample_post_2D_to_1D(post_param_2d):
    ne_1d = Ne_MDA
    ne_2d = Ne_opt
    nx = dims[0]
    nz = dims[2]
    n_params_per_type = nz
    param_vec_1d_list = []
    for post_param_one_ens_member in post_param_2d:
        # reshaped = post_param_one_ens_member.reshape(2, dims[0]*dims[2])
        reshaped = post_param_one_ens_member.reshape(2, nx, nz, order="C")
        #  use results from 2d inversion in column with logging point
        param_vec_1d = reshaped[:, nx // 2, :]
        if not keep_inversion_lim_z:
            param_vec_1d = np.concatenate((param_vec_1d[:, 1:], param_vec_1d[:, -1:]), axis=1)
        param_vec_1d = param_vec_1d.reshape(-1, order="C")
        param_vec_1d_list.append(param_vec_1d)

    arr = np.vstack(param_vec_1d_list)  # shape (n_vectors, L)
    #Alternatives: use mean or best estimate after previous 2D inversion as prior
    best_idx = np.argmin(post_loss)
    prior_param_vec_1d = arr[best_idx,:]  # shape (L,)
    #prior_param_vec_1d = arr.mean(axis=0)  # shape (L,)

    theta_1d = np.empty((2 * n_params_per_type, ne_1d), dtype=float)
    cov_blocks = []
    mean_blocks = []

    # set up rh mean and std used for making covariance
    normal_rh_mean_0d = float(np.median(prior_param_vec_1d[:n_params_per_type]))
    if apply_bounds:
        normal_rh_std_1d = normal_bounded_std_from_lognormal_std(log_normal_rh_std_target, 0)
    else:
        normal_rh_std_1d = normal_std_from_lognormal_std(log_normal_rh_std_target, normal_rh_mean_0d)
    #
    oneD_std = [normal_rh_std, ratio_latent_std]  # Standard deviations for the 1D fields, can be tuned

    for param_idx in range(2):
        start = param_idx * n_params_per_type
        stop = start + n_params_per_type
        prior_vec = (prior_param_vec_1d[start:stop])  # np.full(nz, stationary_mean, dtype=float)
        stationary_std = float(oneD_std[param_idx])
        mean_blocks.append(prior_vec)
        cov = geostat.gen_cov2d(nz, 1, stationary_std ** 2, var_range_cells, 1, 0, 'exp')
        if apply_bounds:
            samples = apply_tighter_bounds_for_prior_ensemble(n_params_per_type, ne_1d, param_idx, cov, prior_vec)
        else:
            samples = geostat.gen_real(prior_vec, cov, ne_1d).reshape(n_params_per_type, ne_1d, order='C')

        theta_1d[start:stop, :] = samples
        cov_blocks.append(cov)
    c_theta_1d = np.block([[cov_blocks[0], np.zeros_like(cov_blocks[0])],
                           [np.zeros_like(cov_blocks[1]), cov_blocks[1]], ])
    mean_theta_1d = np.mean(theta_1d, axis=1)


    return theta_1d, c_theta_1d, mean_theta_1d

def apply_tighter_bounds_for_prior_ensemble(n_params_per_type, n_needed, param_idx, cov, mean_vec):

    batch_size = max(2 * n_needed, 100)
    samples = np.empty((n_params_per_type, 0))
    rng = np.random.default_rng()
    max_attempts = 100
    attempts = 0
    current_batch = batch_size

    if param_idx == 0:
        min_value = latent_rh_min
        max_value = latent_rh_max
    else:
        min_value = latent_ratio_min
        max_value = latent_ratio_max

    # simple rejection loop
    while samples.shape[1] < n_needed and attempts < max_attempts:
        z = geostat.gen_real(mean_vec, cov, batch_size)  # generate batch of latents; shape (nz, batch_size)
        mask_valid = np.all((z > min_value) & (z < max_value), axis=0)
        if mask_valid.any():
            keep = z[:, mask_valid]
            samples = np.concatenate((samples, keep), axis=1)
        attempts += 1
        # adapt: increase batch if acceptance low
        if attempts % 5 == 0:
            current_batch = min(current_batch * 2, 10000)

    if samples.shape[1] >= n_needed:
        return samples[:, :n_needed]

    marg_std = np.sqrt(np.diag(cov))
    z_fallback = np.empty((n_params_per_type, n_needed))
    a = (min_value - mean_vec) / marg_std
    b = (max_value - mean_vec) / marg_std
    for i in range(n_params_per_type):
        # draw independent truncated normals per location
        z_fallback[i, :] = truncnorm.rvs(a[i], b[i], loc=mean_vec[i], scale=marg_std[i],
                                         size=n_needed, random_state=rng)
    return z_fallback

def _build_param_dict(param):
    """Map optimizer parameters to simulator input dictionary."""
    n_params_per_type = dims[0] * dims[2]  # nx * nz
    param_sim = param
    # Optimizer vector uses (nx, nz) with C-order flattening (z-fastest).
    rh_2d = param_sim[:n_params_per_type].reshape(dims[0], dims[2], order='C')
    rh_ratio_2d = param_sim[n_params_per_type:].reshape(dims[0], dims[2], order='C')
    
    # make rh -remove bounds if optimizer is bounded
    if apply_bounds:  # param is m_bounded
        rh_2d_unbounded = remove_bounds_rh(rh_2d)
    else:
        rh_2d_unbounded = rh_2d
    # Broadcast 2D arrays to 3D by tiling along the y-axis
    rh = np.tile(rh_2d_unbounded[:, np.newaxis, :], (1, dims[1], 1))

    # make rv - using reciprocal distribution for ratio
    rh_ratio = np.tile(rh_ratio_2d[:, np.newaxis, :], (1, dims[1], 1))
    u = norm.cdf(rh_ratio)  # Map to [0, 1]
    ratio = min_ratio * np.power(max_ratio / min_ratio, u)  # Reciprocal distribution
    rv = np.log(np.exp(rh) * ratio)
    

    param_dict = {'rh': rh.flatten(order='F'),
                  'rv': rv.flatten(order='F')
        }
    return param_dict, rh_2d, rh_ratio_2d

def _build_param_dict_1d(param):
    """Map a z-column parameter vector to the 1D simulator input dictionary."""
    n_params_per_type = dims[2]
    param_sim = param

    if np.size(param_sim) != 2 * n_params_per_type:
        raise ValueError(
            f"Expected a 1D state vector of length {2 * n_params_per_type}, got {np.size(param)}"
        )
    rh_1d = np.asarray(param_sim[:n_params_per_type], dtype=float)
    rh_ratio_1d = np.asarray(param_sim[n_params_per_type:], dtype=float)

    # make rh -remove bounds if optimizer is bounded
    if apply_bounds:  # param is m_bounded
        rh_1d_unbounded = remove_bounds_rh(rh_1d)
    else:
        rh_1d_unbounded = rh_1d

    u = norm.cdf(rh_ratio_1d)
    ratio = min_ratio * np.power(max_ratio / min_ratio, u)
    rv_1d = np.log(np.exp(rh_1d_unbounded) * ratio)

    param_dict = {'rh': rh_1d_unbounded,
                  'rv': rv_1d
        }
    return param_dict, rh_1d, rh_ratio_1d

def _get_1d_surface_depth(logging_point_idx):
    """Return a column whose center cell is aligned with the current TVD."""
    column_top_ft = TVD[logging_point_idx] - 0.5 * dims[2] * dZ
    return [column_top_ft + i * dZ for i in range(dims[2])]

def _set_logging_point_state_0d(logging_point_idx):
    """Update tool/grid state for current logging point."""
    UTA0D_sim.tool['tvd']  = TVD[logging_point_idx:logging_point_idx + 1]
    UTA0D_sim.tool['MD'] =  MD[logging_point_idx:logging_point_idx + 1]
    UTA0D_sim.tool['X'] = WX[logging_point_idx:logging_point_idx + 1]
    UTA0D_sim.tool['ijk'] =  WY[logging_point_idx:logging_point_idx + 1]
    #UTA0D_sim.tool['nlog'] = 1

def _set_logging_point_state_1d(logging_point_idx, keep_inversion_z_lim = None):
    """Update tool/grid state for current logging point."""
    UTA1D_sim.tool['tvd'] = TVD[logging_point_idx:logging_point_idx + 1]
    UTA1D_sim.tool['MD'] = MD[logging_point_idx:logging_point_idx + 1]
    UTA1D_sim.tool['X'] = WX[logging_point_idx:logging_point_idx + 1]
    UTA1D_sim.tool['ijk'] = WY[logging_point_idx:logging_point_idx + 1]

    if el == 0 or keep_inversion_z_lim is None:
        UTA1D_sim.tool['surface_depth'] = _get_1d_surface_depth(logging_point_idx)
    else:
        if not keep_inversion_z_lim: # shift by one grid cell
            UTA1D_sim.tool['surface_depth'] = [depth + dZ for depth in UTA1D_sim.tool['surface_depth']]

def _set_logging_point_state_2d(logging_point_idx, keep_inversion_z_lim = None, keep_inversion_x_lim = None):
    """Update tool/grid state for current logging point."""

    UTA2D_sim.tool['tvd'] = TVD[logging_point_idx:logging_point_idx + 1]
    UTA2D_sim.tool['MD'] = MD[logging_point_idx:logging_point_idx + 1]
    UTA2D_sim.tool['X'] = WX[logging_point_idx:logging_point_idx + 1]
    UTA2D_sim.tool['ijk'] = WY[logging_point_idx:logging_point_idx + 1]


    well_x_m = wellpath['X'][logging_point_idx][0]
    well_y_m = wellpath['Y'][logging_point_idx][0]
    well_z_m = wellpath['Z'][logging_point_idx][0]

    if el == 0 or keep_inversion_z_lim is None:
        UTA2D_sim.model['shift'] = {'x': well_x_m * meter_to_feet - dX * (dims[0]//2),
                                'y': well_y_m * meter_to_feet - dY * (dims[1]//2),
                                'z': well_z_m * meter_to_feet - dZ * (dims[2]//2)}
    else:

        if not keep_inversion_z_lim:
            UTA2D_sim.model['shift']['z'] += dZ
        if not keep_inversion_x_lim:
            UTA2D_sim.model['shift']['x'] +=  dX
        UTA2D_sim.model['shift']['y'] = well_y_m * meter_to_feet - dY * (dims[0] // 2)

def simulate_pred_only(param, simfidelity="2D"):
    """Forward response only (no Jacobian expected)."""
    if simfidelity == "0D":
        param_sim = np.zeros_like(param)
        if apply_bounds:  # param is now m_bounded
            param_sim[0] = remove_bounds_rh(param[0])
            param_sim[1] = param[1]
        else:
            param_sim = param
        param_dict = {'rh':param_sim[0],
                      'rv':np.log(np.exp(param_sim[0]) *
                                  (min_ratio*np.power(max_ratio / min_ratio, norm.cdf(param[1]))))}
        #_set_logging_point_state(logging_point_idx, keep_inversion_lim_z, keep_inversion_lim_x)
        pred = UTA0D_sim.run_fwd_sim(param_dict, 0)
        pred_vec = np.concatenate([pred[0][k][selected_data_indices_0D] for k in tools])

    elif simfidelity == "1D":
        param_dict, _, _ = _build_param_dict_1d(param)
        #_set_logging_point_state(logging_point_idx, keep_inversion_lim_z, keep_inversion_lim_x)
        pred = UTA1D_sim.run_fwd_sim(param_dict, 0)
        pred_vec = np.concatenate([pred[0][k][selected_data_indices_1D] for k in tools])
        
    else:        
        param_dict, _, _ = _build_param_dict(param)
        #_set_logging_point_state(logging_point_idx, keep_inversion_lim_z, keep_inversion_lim_x)
        original_jac_flag = UTA2D_sim.options.get('jacobi', True)
        UTA2D_sim.options['jacobi'] = False
        try:
            pred = UTA2D_sim.run_fwd_sim(param_dict, 0)
        finally:
            UTA2D_sim.options['jacobi'] = original_jac_flag
        pred_vec = np.concatenate([pred[0][k][selected_data_indices] for k in tools])
    return pred_vec

def simulate_and_grad(param):
    param_dict, rh_2d, rh_ratio_2d = _build_param_dict(param)
    
    #_set_logging_point_state(logging_point_idx, keep_inversion_z_lim = keep_inversion_lim_x, keep_inversion_x_lim = keep_inversion_lim_x)
    
    pred,jacobian = UTA2D_sim.run_fwd_sim(param_dict, 0)
    
    jac_list = []
    for param_idx, param_key in enumerate(['rh', 'rv']):
        # Collect Jacobian for this parameter across all tool settings in the same order as data_keys
        jac_param_list = []
        for tool_key in tools:
            # jacobian block layout from EMsim is (ncomp, nz, ny, nx).
            # After summing broadcasted y-direction we get (ncomp, nz, nx).
            # Optimizer parameter layout is (nx, nz) flattened with C-order (z-fastest).
            # Reorder Jacobian from (nz, nx) -> (nx, nz) before flattening.
            jac_tool_selected = np.sum(jacobian[0][tool_key][param_idx], axis=2)[selected_data_indices, :, :]
            jac_tool_selected = jac_tool_selected.transpose(0, 2, 1)  # (ncomp, nx, nz)
            jac_param_list.append(jac_tool_selected.reshape(jac_tool_selected.shape[0], -1))
        # Concatenate along data axis: stack data from all tool settings
        jac_param = np.vstack(jac_param_list)
        jac_list.append(jac_param)

    # -------------------------------------------------------------------------
    # Chain rule from Fortran Jacobian space -> optimizer space.
    #
    # jac_list[0] : d(data)/dRh_2D   (after collapsing broadcasted y-direction)
    # jac_list[1] : d(data)/dRv_2D   (after collapsing broadcasted y-direction)
    #
    # Optimizer parameters:
    #   m = log(Rh_2D)
    #   m_bounded = log((Rh_2D-Rh_min)/(Rh_max-Rh_2D))
    #   z = Gaussian latent for anisotropy ratio
    #   ratio = min_ratio * np.power(max_ratio / min_ratio, u_2d)
    #   Rv_2D = Rh_2D * ratio
    #
    # Needed derivatives:
    #   dRh/dm = Rh
    #   dRh/dm_bounded = (Rh-Rh_min)*(Rh_max-Rh)/(Rh_max-Rh_min)
    #   dRv/dm = Rv
    #   dRv/dm_b = (rh - rh_min) * (rh_max - rh) / (rh_max - rh_min)*ratio
    #   dRv/dz = Rh * (max_ratio-min_ratio) * phi(z)
    #
    # Therefore:
    #   d(data)/dm = d(data)/dRh * dRh/dm + d(data)/dRv * dRv/dm
    #   d(data)/dm_bounded = d(data)/dRh * dRh/dm_bounded + d(data)/dRv * dRv/dRh * dRh/dm_bounded
    #   d(data)/dz = d(data)/dRv * dRv/dz
    # -------------------------------------------------------------------------
    j_rh = jac_list[0]  # d(data)/dRh_2D
    j_rv = jac_list[1]  # d(data)/dRv_2D

    u_2d = norm.cdf(rh_ratio_2d)
    ratio_2d = min_ratio * np.power(max_ratio / min_ratio, u_2d)  # Reciprocal distribution
    dratio_dz_2d = np.log(max_ratio / min_ratio) * ratio_2d * norm.pdf(rh_ratio_2d)
    if apply_bounds:
        rh_2d_unbounded = remove_bounds_rh(rh_2d)
    else:
        rh_2d_unbounded = rh_2d

    rh_phys_2d = np.exp(rh_2d_unbounded)
    rv_phys_2d = rh_phys_2d * ratio_2d

    # Keep flatten convention consistent with reshape(...).reshape(-1) above.
    rh_phys_vec = rh_phys_2d.reshape(-1, order='C')
    rv_phys_vec = rv_phys_2d.reshape(-1, order='C')
    drv_dz_vec = (rh_phys_2d * dratio_dz_2d).reshape(-1, order='C')
    drh_dm_bounded_2d = (rh_phys_2d - rh_phys_min)*(rh_phys_max- rh_phys_2d)/(rh_phys_max-rh_phys_min)
    drv_drh = ratio_2d
    drv_dm_bounded_2d = drv_drh * drh_dm_bounded_2d
    drh_dm_bounded_vec = drh_dm_bounded_2d.reshape(-1, order='C')
    drv_dm_bounded_vec = drv_dm_bounded_2d.reshape(-1, order='C')
    if apply_bounds: # m = m_bounds
        j_m = j_rh * drh_dm_bounded_vec[np.newaxis, :] + j_rv * drv_dm_bounded_vec[np.newaxis, :]
    else:
        j_m = j_rh * rh_phys_vec[np.newaxis, :] + j_rv * rv_phys_vec[np.newaxis, :]
    j_z = j_rv * drv_dz_vec[np.newaxis, :]

    # Jacobian wrt optimizer parameters [m, z]
    jacobian_matrix = np.hstack([j_m, j_z])
    jacobian_matrix_phys = np.hstack([j_rh, j_rv])
    pred_vec = np.concatenate([pred[0][k][selected_data_indices] for k in tools])

    return pred_vec, jacobian_matrix, jacobian_matrix_phys #, hessian_approx

def custom_loss(predictions, data_real, theta, theta_mean, C_theta, Cd):
    # Example: weighted mean squared error
    residuals_data = data_real - predictions.flatten()
    data_loss = 0.5 * np.sum((residuals_data ** 2) / Cd)
    
    residuals_theta = theta.flatten() - theta_mean.flatten()
    
    theta_loss = 0.5 * residuals_theta @ np.linalg.solve(C_theta, residuals_theta)

    loss = data_loss #+ theta_loss
    
    return loss

def map_update_data_space_LM(theta, f_theta, theta_mean, J, y, Cd_diag, C_theta, step_lambda):
    """
    Levenberg-Marquardt MAP update in data-space using matrix inversion lemma.
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
    theta_prior = theta_mean.flatten()
    f_theta = f_theta.flatten()
    y = y.flatten()
    Cd_diag = Cd_diag.flatten()

    # Residuals in model and data spaces.
    r_d = f_theta - y
    damp_scale = 1.0 + step_lambda
    r_m = (theta - theta_prior) / damp_scale

    # Data-space MAP Levenberg-Marquardt step:
    #   delta = -r_m - C_theta J^T [ (1+lambda)Cd + J C_theta J^T ]^{-1} (r_d - J r_m)
    # 1: H_data^{-1} = [ (1+lambda)Cd + J C_theta J^T ]^{-1}
    # 2. CM_JT = C_theta @ J.T
    # 3: intermediate  = C_theta J^T @ H_data^{-1} (nm,nd)
    # 4: første rad av intermediate *første verdi av rhs (oppdatering av modell for første data punkt
    # 5: plot disse individuelt (nd plot som J)
    CM_JT = C_theta @ J.T
    H_data = damp_scale * np.diag(Cd_diag) + J @ CM_JT
    rhs = r_d - J @ r_m

    try:
        w = np.linalg.solve(H_data, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(H_data, rhs, rcond=None)[0]

    delta_theta = -r_m - CM_JT @ w
    return delta_theta

def disallow_updates_prior_to_bit(search_dir_orig, boundary_layer = 0):
    search_dir_t = search_dir_orig.T
    for param_indx in range(2):
        start = dims[2] * dims[0] * param_indx
        stop = dims[2] * dims[0] * (param_indx + 1)
        block = search_dir_t[start:stop]
        reshaped_block = block.reshape(dims[0], dims[2], order="C")
        reshaped_block[:dims[0] // 2, :] = 0
        if boundary_layer > 0:
            reshaped_block[:, :boundary_layer] = 0
            reshaped_block[:, dims[2] - boundary_layer:] = 0

        # put modified block back, preserving original shape and order
        search_dir_t[start:stop] = reshaped_block.reshape(block.shape, order="C")
        # write back updated search_dir (undo transpose)
    search_dir_updated = search_dir_t.T
    return search_dir_updated

def order_1d_ensemble_members_after_loss(top_en_n):
    losses = []
    indices = []
    for i in range(Ne):
        curr_loss_en = custom_loss(pred_curr[:, i], data_real[:, i], param_vec[:, i], mean_theta, c_theta, Cd_vec)
        losses.append(curr_loss_en)
        indices.append(i)

    losses = np.array(losses)
    indices = np.array(indices)
    # Get indices sorted by loss (smallest to largest)
    sorted_indices = indices[np.argsort(losses)]
    # Select the top N members with the smallest losses
    best_members_indices = sorted_indices[:top_en_n]

    return best_members_indices

def make_param_vec_list():
    # list. Length of list defines how many columns (x-values) that is being used in th kriging
    # n_1D_columns: Length of list defines how many columns (x-values) that is being used in th kriging
    # param_1D_columns: indices of columns that is to be used in kriging
    param_vec_list = []
    nz = dims[2]
    # column_idx_no, column_idx in enumerate(param_1D_columns):
    best_members_indices = order_1d_ensemble_members_after_loss(Ne_opt)
    for idx in param_1D_columns:
        param_vec_1d = np.zeros((nz * 2, Ne_opt))
        if idx < param_1D_columns[-1]:
            for ne_member, post_param_one_ens_member in enumerate(post_param):
                reshaped = post_param_one_ens_member.reshape(2, dims[0] * dims[2])
                for param_idx in range(2):  # number of parameters (rh, rv)
                    param_vec_1d[param_idx*nz:(param_idx+1)*nz, ne_member] = reshaped[param_idx, idx* nz: (idx+1)* nz]
        elif idx == param_1D_columns[-1]:
            param_vec_1d = param_vec[:, best_members_indices]
        param_vec_list.append(param_vec_1d)

    return param_vec_list

def plot_ensemble_predictions(pred_curr, data_real, data_vec, title_suffix='', filename='ensemble_predictions.png'):
    """Plot ensemble predictions vs observed data with percentile bands.

    Parameters:
    pred_curr: [d, Ne] - ensemble predictions
    data_real: [d, Ne] - perturbed observed data
    data_vec: [d] - true observed data
    title_suffix: str - suffix for plot title
    filename: str - output filename
    """
    plt.figure(figsize=(10, 6))
    x_indices = np.arange(pred_curr.shape[0])

    # Compute percentiles for pred_curr (axis=1 is ensemble dimension)
    pred_p05 = np.percentile(pred_curr, 5, axis=1)
    pred_p95 = np.percentile(pred_curr, 95, axis=1)
    pred_median = np.percentile(pred_curr, 50, axis=1)

    # Compute percentiles for data_real
    data_p05 = np.percentile(data_real, 5, axis=1)
    data_p95 = np.percentile(data_real, 95, axis=1)
    data_median = np.percentile(data_real, 50, axis=1)

    # Plot pred_curr first
    plt.fill_between(x_indices, pred_p05, pred_p95, alpha=0.3, color='blue', label='Pred 5-95%')
    plt.plot(x_indices, pred_median, 'b-', label='Pred median')

    # Plot data_real after
    plt.fill_between(x_indices, data_p05, data_p95, alpha=0.3, color='red', label='Data 5-95%')
    plt.plot(x_indices, data_median, 'r--', label='Data median')

    plt.plot(data_vec, 'kx', label='True Data', markersize=10)
    plt.xlabel('Data Point Index')
    plt.ylabel('Data Value')
    plt.title(f'Ensemble Predictions vs Observed Data{title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_distributions_using_seaborn_rv():
    import seaborn as sns
    ne = 1000
    # A target standard deviation for rv?
    log_normal_std_target = 3
    rh_log_values = [0.5, 1, 3]

    ratio_latent_mean_values = [-1, -0.5, 0, 0.5, 1]

    #
    ratio_latent_std_plot = 1

    # Plot using seaborn


    for rh in rh_log_values:
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        # Generate ratio latent samples
        for ratio_latent_mean in ratio_latent_mean_values:
            ratio_latent_samples = np.random.normal(loc=ratio_latent_mean, scale=ratio_latent_std_plot, size=(ne,))

            sns.kdeplot(ratio_latent_samples, fill=True, color="skyblue", ax=axs[0],
                        label=f"ratio_latent\nMean: {ratio_latent_mean:.1f}, Std: {ratio_latent_std_plot:.1f}")
            axs[0].axvline(ratio_latent_mean, color='red', linestyle='--')

            # Convert to rv samples
            u = norm.cdf(ratio_latent_samples)  # Map to [0, 1]
            ratio = min_ratio * np.power(max_ratio / min_ratio, u)  # Reciprocal distribution
            ratio_mean = np.mean(ratio)
            ratio_std = np.std(ratio)
            # Plot ratio samples
            sns.kdeplot(ratio, fill=True, color="green", ax=axs[1],
                        label=f"Mean: {ratio_mean:.1f}, Std: {ratio_std:.1f}")
            axs[1].axvline(ratio_mean, color='red', linestyle='--')

            rv_samples = (np.exp(rh) * ratio)
            rv_mean = np.mean(rv_samples)
            rv_std = np.std(rv_samples)
            # Plot rv_samples
            sns.kdeplot(rv_samples, fill=True, color="green", ax=axs[2],
                        label=f"Mean: {rv_mean:.1f}, Std: {rv_std:.1f}")
            axs[2].axvline(rv_mean, color='red', linestyle='--')

        axs[0].set_title(f'Density plot for ratio latent samples (normal distribution)')
        axs[0].set_xlabel('Value')
        axs[0].set_ylabel('Density')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].set_title(f'Density plot for the rh ratio samples')
        axs[1].set_xlabel('Value')
        axs[1].set_ylabel('Density')
        axs[1].legend()
        axs[1].grid(True)

        axs[2].set_title(f'Density plot for rv samples (xx distribution); rh: {np.exp(rh):.1f}')
        axs[2].set_xlabel('Value')
        axs[2].set_ylabel('Density')
        axs[2].legend()
        axs[2].grid(True)

    plt.tight_layout()
    plt.show()

def plot_distributions_using_seaborn_rh():
    import seaborn as sns
    ne = 1000
    # normal distribution fro log-param-values
    normal_mean_values = [0.5, 1, 1.5, 2, 3]

    if apply_bounds:
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    # Generate samples
    for normal_mean in normal_mean_values:
        normal_rh_std_plot = normal_std_from_lognormal_std(log_normal_rh_std_target, normal_mean)
        #log_rh_std  = find_std_keeping_variance_fixed(log_rh_mean)
        #variance_rh_samples = (np.exp(log_rh_std ** 2) - 1) * np.exp(2 * log_rh_mean + log_rh_std ** 2)
        normal_rh_samples = np.random.normal(loc=normal_mean, scale=normal_rh_std_plot, size=(ne,))
        #ratio_latent_samples = np.random.normal(loc=ratio_latent_mean, scale=ratio_latent_std, size=(ne,))
        # Plot log_rh_samples
        sns.kdeplot(normal_rh_samples, fill=True, color="skyblue", ax=axs[0], label=f"log_rh_samples\nMean: {normal_mean:.1f}, Std dev: {normal_rh_std_plot:.1f}")

        # Add a vertical line for the mean
        axs[0].axvline(normal_mean, color='red', linestyle='--')

    axs[0].set_title('Density plot for log(rh) samples (normal distribution)')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Density')
    axs[0].legend()
    axs[0].grid(True)
    for normal_mean in normal_mean_values:
        # Generate log_rh_samples
        #log_normal_rh_std = find_std_keeping_variance_fixed(normal_mean, 1, 1.7)
        normal_rh_std_plot = normal_std_from_lognormal_std(log_normal_rh_std_target, normal_mean)
        normal_rh_samples = np.random.normal(loc=normal_mean, scale=normal_rh_std_plot, size=(ne,))
        # Convert to rh_samples
        log_normal_rh_samples = np.exp(normal_rh_samples)
        log_normal_mean, log_normal_rh_std = compute_log_normal_params(normal_mean, normal_rh_std)
        # Plot rh_samples
        sns.kdeplot(log_normal_rh_samples, fill=True, color="green", ax=axs[1], label=f"Mean: {log_normal_mean:.1f}, Std: {log_normal_rh_std:.1f}")

        # Add a vertical line for the mean in log-normal
        axs[1].axvline(log_normal_mean, color='red', linestyle='--')

    # Finalize the second subplot
    axs[1].set_title('Density plot for rh samples (log-normal distribution)')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Density')
    axs[1].legend()
    axs[1].grid(True)

    if apply_bounds:
        for normal_mean in normal_mean_values:
            normal_bounded_mean = apply_bounds_rh(normal_mean)
            normal_bounded_rh_std_plot = normal_bounded_std_from_lognormal_std(log_normal_rh_std_target, normal_mean)
            # log_rh_std  = find_std_keeping_variance_fixed(log_rh_mean)
            # variance_rh_samples = (np.exp(log_rh_std ** 2) - 1) * np.exp(2 * log_rh_mean + log_rh_std ** 2)
            normal_bounded_rh_samples = np.random.normal(loc=normal_bounded_mean, scale=normal_bounded_rh_std_plot, size=(ne,))

            sns.kdeplot(normal_bounded_rh_samples, fill=True, color="skyblue", ax=axs[2],
                        label=f"log_rh_bounded_samples\nMean: {normal_bounded_mean:.1f}, Std: {normal_bounded_rh_std_plot:.1f}")

            # Add a vertical line for the mean
            axs[2].axvline(normal_bounded_mean, color='red', linestyle='--')

        axs[2].set_title('Density plot for log((rh-rh_min)/(rh_max-rh))) samples (normal distribution)')
        axs[2].set_xlabel('Value')
        axs[2].set_ylabel('Density')
        axs[2].legend()
        axs[2].grid(True)


    plt.tight_layout()
    plt.show()

def assess_intermediate_param_values(fidelity_level, post_estim = False):
    if fidelity_level == 0:
        nx = 1
        nz = 1
    elif fidelity_level == 1:
        _, _, nz = dims
        nx = 1
    else:
        nx, _, nz = dims

    if fidelity_level == 2 and post_estim:
        plot_param = np.asarray(post_param)
    else:
        plot_param = np.asarray(param_vec).T
    n_param_per_type = nx * nz

    if apply_bounds: #and fidelity_level == 2 and not post_estim: # param is now m_bounded. Transfer to m = log(rh)
        rh = remove_bounds_rh(plot_param[:,:n_param_per_type].reshape(-1, nx, nz, order="C"))
        rh_prior = remove_bounds_rh(mean_theta[:n_param_per_type].reshape(nx, nz, order="C"))
    else:
        rh = plot_param[:,:n_param_per_type].reshape(-1, nx, nz, order="C")
        rh_prior = mean_theta[:n_param_per_type].reshape(nx, nz, order="C")

    latent_ratio = plot_param[:,n_param_per_type:].reshape(-1, nx, nz, order="C")
    latent_ratio_prior = mean_theta[n_param_per_type:].reshape(nx, nz, order="C")
    u = norm.cdf(latent_ratio)
    u_prior = norm.cdf(latent_ratio_prior)
    ratio = min_ratio * np.power(max_ratio / min_ratio, u)
    ratio_prior = min_ratio * np.power(max_ratio / min_ratio, u_prior)
    rv = (np.exp(rh) * ratio)
    rv_prior = (np.exp(rh) * ratio_prior)
    rh = np.exp(rh)
    rh_prior = np.exp(rh_prior)

    # Covariance for model parameters
    rh_c = np.diag(c_theta)[:n_param_per_type]
    if apply_bounds:
        rh_c = remove_bounds_rh(rh_c)
    rh_c = np.exp(rh_c)
    latent_ratio_c = np.diag(c_theta)[-1]
    u_c = norm.cdf(latent_ratio_c)
    ratio_c = min_ratio * np.power(max_ratio / min_ratio, u_c)



    if post_estim:
        str_part = "after"
    else:
        str_part = "before"
    print(f' ')
    print(f'[min, mean, max] values for ratio {str_part} D{fidelity_level} inversion is: [{np.min(ratio):1f}, {np.mean(ratio):1f}, {np.max(ratio):1f}]')
    print(f'max value for ratio covariance {str_part} D{fidelity_level} inversion is: {np.max(ratio_c):1f}')
    print(f' ')
    print(f'[min, mean, max] values for rh {str_part} D{fidelity_level} inversion is: [{np.min(rh):1f}, {np.mean(rh):1f}, {np.max(rh):1f}]')
    print(f'[min, mean, max] values for rh prior {str_part} D{fidelity_level} inversion is: [{np.min(rh_prior):1f}, {np.mean(rh_prior):1f}, {np.max(rh_prior):1f}]')
    print(f'max value for rh covariance {str_part} D{fidelity_level} inversion is: {np.max(rh_c):1f}')
    print(f' ')
    print(f'[min, mean, max] values for rv {str_part} D{fidelity_level} inversion is: [{np.min(rv):1f}, {np.mean(rv):1f}, {np.max(rv):1f}]')
    print(f'[min, mean, max] values for rv prior {str_part} D{fidelity_level} inversion is: [{np.min(rv_prior):1f}, {np.mean(rv_prior):1f}, {np.max(rv_prior):1f}]')
    #print(f'median value for rh {str_part} D{fidelity_level} inversion is: {rh_median}')
    #print(f'median value for rv {str_part} D{fidelity_level} inversion is: {rv_median}')

def rescale_minmax(X, new_min, new_max):
    X = np.asarray(X, dtype=float)
    old_min = X.min()
    old_max = X.max()
    if old_max == old_min:
        # all values identical -> set to new_min (or (new_min+new_max)/2)
        return np.full_like(X, fill_value=new_min)
    if new_min < old_min:
        new_min = old_min
    if new_max > old_max:
        new_max = old_max
    return new_min + (X - old_min) * (new_max - new_min) / (old_max - old_min)

def clip_minmax(X, new_min, new_max):
    X = np.asarray(X, dtype=float)
    return np.clip(X, new_min, new_max)

def smooth_param_field_2d(param_vec, dims_input, sigma, which='rh'):
    """
    Smooth one parameter type in a 2D parameter vector.

    param_vec : 1D array, length = 2 * nx * nz, in order [rh_block, ratio_block]
    dims_input      : (nx, ny, nz)
    sigma     : float or tuple, Gaussian std in grid cells
    which     : 'rh' or 'ratio' (latent ratio)

    Returns a new 1D param_vec with the chosen block smoothed.
    """
    nx, ny, nz = dims_input
    n_params_per_type = nx * nz

    out = param_vec.copy()

    if which == 'rh':
        start, stop = 0, n_params_per_type
    elif which == 'ratio':
        start, stop = n_params_per_type, 2 * n_params_per_type
    else:
        raise ValueError("which must be 'rh' or 'ratio'")

    block = out[start:stop].reshape(nx, nz, order='C')

    # If using bounds for rh, better smooth in "unbounded" log‑rh:
    if apply_bounds and which == 'rh':
        block = clip_minmax(block, latent_rh_min, latent_rh_max)
        block_unbounded = remove_bounds_rh(block)
        block_unbounded_s = gaussian_filter(block_unbounded, sigma=sigma, mode='nearest')

    else:
        block = clip_minmax(block, latent_ratio_min, latent_ratio_max)
        block_s = gaussian_filter(block, sigma=sigma, mode='nearest')


    out[start:stop] = block_s.reshape(-1, order='C')
    return out

def apply_bounds_rh(unbounded_param):
    numerator = np.exp(unbounded_param) - rh_phys_min
    denominator = rh_phys_max - np.exp(unbounded_param)
    eps = max(np.finfo(float).eps, 1e-2 * max(np.abs(numerator).max(), np.abs(denominator).max()))
    bounded_param = np.log(np.clip(numerator, eps, None)) - np.log(np.clip(denominator, eps, None))
    return bounded_param

def remove_bounds_rh(bounded_param):
    m_b_exp = np.exp(bounded_param)
    rh_physical = (m_b_exp * rh_phys_max + rh_phys_min) / (1 + m_b_exp)
    unbounded_param = np.log(rh_physical)
    return unbounded_param

def impose_boundary_layer_1d(param_input, rh_shale_phys = 1.5, rv_shale_phys = 4):
    n_params_per_type = dims[2]

    rh_1d = np.asarray(param_input[:n_params_per_type,:], dtype=float)
    rh_ratio_1d = np.asarray(param_input[n_params_per_type:,:], dtype=float)

    # apply bounds if optimizer is bounded
    if apply_bounds:  # param is m_bounded
        rh_shale = apply_bounds_rh(np.log(rh_shale_phys))
    else:
        rh_shale = np.log(rh_shale_phys)
    #impose boundary layer
    rh_1d[:no_fixed_cells_z,:] = rh_shale
    rh_1d[n_params_per_type -no_fixed_cells_z :, :] = rh_shale

    # compute ratio mapping in optimization space,
    ratio_shale = rv_shale_phys / rh_shale_phys
    denom = np.log(max_ratio / min_ratio)
    u = np.log(ratio_shale / min_ratio) / denom
    u_clip = np.clip(u, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    rh_ratio_shale = norm.ppf(u_clip)

    # impose boundary layer
    rh_ratio_1d[:no_fixed_cells_z, :] = rh_ratio_shale
    rh_ratio_1d[n_params_per_type - no_fixed_cells_z:, :] = rh_ratio_shale
    param_output = param_input
    param_output[:n_params_per_type, :] = rh_1d
    param_output[n_params_per_type:, :] = rh_ratio_1d

    return param_output

restart_simulations = False

log_normal_rh_std_target = 3.
normal_rh_mean = 1.5
normal_rh_bounded_mean = apply_bounds_rh(normal_rh_mean)
normal_rh_std = normal_std_from_lognormal_std(log_normal_rh_std_target, normal_rh_mean)
normal_rh_bounded_std = normal_bounded_std_from_lognormal_std(log_normal_rh_std_target, normal_rh_bounded_mean)
ratio_latent_mean = 0
ratio_latent_std = 0.5
if apply_bounds:
    latent_rh_min = apply_bounds_rh(np.log(rh_phys_min*1.25))
    latent_rh_max = apply_bounds_rh(np.log(rh_phys_max*0.65))
    latent_ratio_min = -0.5
    latent_ratio_max = 0.5
    pr, c_theta, mean_theta = sample_prior_0D(Ne, normal_rh_bounded_mean, normal_rh_bounded_std)
else:
    pr, c_theta, mean_theta = sample_prior_0D(Ne, normal_rh_mean, normal_rh_std)

UTA0D_sim, UTA1D_sim, UTA2D_sim, TVD, MD, WX, WY, wellpath = setup_simulators(reference_model_path, 'Bfield')

# Load reference model for plotting
with h5py.File(reference_model, "r") as f:
    ref_model = h5_to_dict(f)

# initalize the parameter vector for optimization
param_vec = copy.deepcopy(pr)

#tot_assim_index = [180]
tot_assim_index = [[el] for el in range(250)]
datatype = 'Bfield'

data = pd.read_pickle('data.pkl')
Cd = pd.read_pickle('var.pkl')

# setup ESMDA update
at = approx_update()
at.proj = (np.eye(Ne) - np.ones((Ne, Ne))/Ne) / np.sqrt(Ne - 1)
at.trunc_energy = 0.95
at.keys_da = {}
#{'localization':{"autoadaloc":0.5,
#                              "type": "sigm",
#                              "field": [dims[0], dims[2]]}
#        }
#at.localization = localization(
#                    at.keys_da['localization'],
#                    [],
#                    [],
#                    [],
#                    Ne
#                )
at.list_states = ['rh', 'rv']
at.prior_info = {'rh':{"active": np.prod([dims[0], dims[2]])},
                 'rv':{"active": np.prod([dims[0], dims[2]])}
        }
at.state_scaling = np.ones(param_vec.shape[0]) # No scaling for now, but could be tuned for better performance
at.lam = 0

#plot_distributions_using_seaborn_rh()
#plot_distributions_using_seaborn_rv()
assim_step_integer = 1
post_param = []

for el, assim_index in enumerate(tot_assim_index[20:200:assim_step_integer]):#enumerate([tot_assim_index[0]]):
    Cd_row = Cd.iloc[assim_index[0]]
    Cd_vec = np.concatenate([np.array(Cd_row[ast.literal_eval(tool)][1])[selected_data_indices] for tool in tools])
    data_vec = np.concatenate([data.iloc[assim_index[0]][ast.literal_eval(tool)][selected_data_indices] for tool in tools])
    threshold = len(data_vec)*2
    MDA_inflation_factor = [10.0]
    curr_loss = np.inf
    mda_counter = 0
    mda_inflation = 1/(1 - 1/MDA_inflation_factor[-1]) # to ensure a geometric series

    # initialize 0D representation based on 2D results
    # 1.) find param value in logging point from 2D array
    keep_inversion_lim_z, keep_inversion_lim_x =  logging_point_vs_inversion_domain()

    Ne = Ne_MDA

    if el == 0 and not restart_simulations:
        _set_logging_point_state_0d(assim_index[0])
        #assess_intermediate_param_values(0)
        print(f"Optimizing in 0D at assimilation step {assim_index[0]} with MDA method")
        while curr_loss > threshold and sum([1 / mda for mda in MDA_inflation_factor[:-1]]) < 0.25:
            #print(f"MDA iteration {mda_counter}, inflation factor: {MDA_inflation_factor[-1]:.3f}")
            en_pred = []
            for ne_count in range(Ne):
                en_pred.append(simulate_pred_only(param_vec[:, ne_count], simfidelity="0D"))

            Cd_inflated = Cd_vec * MDA_inflation_factor[-1]
            at.scale_data = np.sqrt(Cd_inflated)
            data_real = np.random.normal(loc=data_vec[:, np.newaxis],
                                         scale=at.scale_data[:, np.newaxis],
                                         size=(len(data_vec), Ne))

            pred_curr = np.array(en_pred).T  # shape (d, Ne)

            if not debug:
                plot_ensemble_predictions(
                    pred_curr, data_real, data_vec,
                    title_suffix=f' after {mda_counter} MDA iterations',
                    filename=f'ensemble_predictions_after_MDA_iter_{mda_counter}.png'
                )

            at.update(
                enX=param_vec,
                enY=pred_curr,
                enE=data_real
            )

            param_vec += at.step

            curr_loss = custom_loss(pred_curr.mean(axis=1), data_real.mean(axis=1), param_vec.mean(axis=1), mean_theta,
                                    c_theta, Cd_vec)
            print(f"Loss: {curr_loss}")

            mda_counter += 1
            MDA_inflation_factor.append(MDA_inflation_factor[-1] * mda_inflation)
            #print(f"MDA condition check: {sum([1 / mda for mda in MDA_inflation_factor]):.3f}")

        #assess_intermediate_param_values(0, post_estim=True)


    if not restart_simulations or el > 0:
        # Now do 1D ESMDA iterations
        if el == 0:
            pr, c_theta, mean_theta = resample_prior_0D_to_1D(Ne,
                                                              param_vec)  # Resample prior, conditioned on the 0D posterior
            _set_logging_point_state_1d(assim_index[0])
        else:
            # successive assimilation step - use 2d results from previous assimilation step as input
            pr, c_theta, mean_theta = resample_post_2D_to_1D(post_param)
            _set_logging_point_state_1d(assim_index[0], keep_inversion_lim_z)

        param_vec = copy.deepcopy(pr)

        at.state_scaling = np.ones(param_vec.shape[0])  # No scaling for now, but could be tuned for better performance
        print(f"Optimizing in 1D at assimilation step {assim_index[0]} with MDA method")
        # assess_intermediate_param_values(1)
        while curr_loss > threshold and sum([1 / mda for mda in MDA_inflation_factor[:-1]]) < 0.8:  # Half energy for 1D
            # print(f"MDA iteration {mda_counter}, inflation factor: {MDA_inflation_factor[-1]:.3f}")
            en_pred = []
            for ne_count in range(Ne):
                en_pred.append(simulate_pred_only(param_vec[:, ne_count], simfidelity="1D"))

            Cd_inflated = Cd_vec * MDA_inflation_factor[-1]
            at.scale_data = np.sqrt(Cd_inflated)
            data_real = np.random.normal(loc=data_vec[:, np.newaxis],
                                         scale=at.scale_data[:, np.newaxis],
                                         size=(len(data_vec), Ne))

            pred_curr = np.array(en_pred).T  # shape (d, Ne)

            if not debug:
                plot_ensemble_predictions(
                    pred_curr, data_real, data_vec,
                    title_suffix=f' after {mda_counter} MDA iterations',
                    filename=f'ensemble_predictions_after_MDA_iter_{mda_counter}_assim_step{assim_index[0]}.png'
                )

            at.update(
                enX=param_vec,
                enY=pred_curr,
                enE=data_real
            )

            param_vec += at.step
            # impose boundary layer
            if no_fixed_cells_z > 0:
                param_vec =  impose_boundary_layer_1d(param_vec, rh_shale_phys = 1.5, rv_shale_phys = 4)

            curr_loss = custom_loss(pred_curr.mean(axis=1), data_real.mean(axis=1), param_vec.mean(axis=1), mean_theta,
                                    c_theta, Cd_vec)
            print(f"Loss: {curr_loss}")

            mda_counter += 1
            MDA_inflation_factor.append(MDA_inflation_factor[-1] * mda_inflation)
            # print(f"MDA condition check: {sum([1/mda for mda in MDA_inflation_factor]):.3f}")

    #assess_intermediate_param_values(1, post_estim=True)
    # Finally, do 2D gradient-based optimization for each ensemble member with the last inflation factor if not converged
    if curr_loss > 0:
        # Resample the prior again, conditioned on the 1D posterior or last 2d if restarting simulations
        if el == 0:
            if restart_simulations:
                ### option to restart simulations at a given assimilation step
                with open(f'inversion_results_assim_{assim_index[0]}.pkl', 'rb') as f:
                    results_restart = pickle.load(f)
                pr = results_restart.get("prior_params")
                mean_theta = np.asarray(results_restart.get("prior_mean_rml"))
                c_theta = np.asarray(results_restart.get("prior_covariance_rml"))
                pr[:dims[0] * dims[2], :] = apply_bounds_rh(pr[:dims[0] * dims[2], :])
                param_vec = copy.deepcopy(pr)
            else:
                n_1D_columns = 1  # dims[0] //2  + 1 # Set the desired number of copies
                param_1D_columns = np.arange(n_1D_columns)
                param_vec_list = [param_vec.copy() for _ in range(n_1D_columns)]
                best_indices = order_1d_ensemble_members_after_loss(Ne_opt)
                pr, c_theta, mean_theta = resample_prior_1D_to_2D(Ne, np.squeeze(param_vec_list, axis=0))#, param_1D_columns)
                param_vec = copy.deepcopy(pr[:, best_indices])  # Take the Ne samples with the lowes loss in case resampling returns more
            _set_logging_point_state_2d(assim_index[0])
        else:
            # assumes logging point moves one grid cell in x-direction per assimilation_step
            # TODO make sure this is correct also for nx being an even number
            n_1D_columns = dims[0] // 2 + 1
            param_1D_columns = np.arange(n_1D_columns)
            if not keep_inversion_lim_x:  # inversion grid is shifted one column since last logging point
                param_1D_columns += 1
            # Create N copies of the array gathered in a list
            param_vec_list = make_param_vec_list()
            pr, c_theta, mean_theta = resample_prior_1D_to_2D_simplified(param_vec_list, param_1D_columns)
            param_vec = copy.deepcopy(pr[:, :Ne_opt]) #
            _set_logging_point_state_2d(assim_index[0], keep_inversion_lim_z, keep_inversion_lim_x)

        #assess_intermediate_param_values(2)
        Ne = min(Ne, Ne_opt)

        final_factor = 1/(1-sum([1/mda for mda in MDA_inflation_factor[:-1]]))
        MDA_inflation_factor[-1] = final_factor # Final inflation factor to ensure proper weighting of data and prior in the last iteration
        # Optimize each ensemble member with gradient-based method
        failed_indices = []
        success_indices = []
        post_param = []
        post_pred = []
        post_loss = []
        post_jac = []
        post_jac_phys = []

        Cd_inflated = Cd_vec * MDA_inflation_factor[-1]
        data_real = np.random.normal(loc=data_vec[:, np.newaxis],
                                    scale=np.sqrt(Cd_inflated)[:, np.newaxis], 
                                    size=(len(data_vec), Ne))



        for ne_count in range(Ne):
            print(
                f"Optimizing ensemble member {ne_count + 1}/{Ne} at assimilation step {assim_index[0]} with gradient-based method")

            param_current = param_vec[:, ne_count]
            data_current = data_real[:, ne_count]

            # initial settings
            tot_loss = []
            lambda_down = 10.0
            lambda_up = 5.0
            lambda_min = 1e-7
            lambda_max = 1e16
            max_lm_trials = 2
            max_iterations = 8

            # Robust initial simulate_and_grad
            try:
                pred_curr, jac_curr, jac_curr_phys = simulate_and_grad(param_current)
            except Exception as e:
                # free large objects, force GC, and log memory before retry
                # delete any large references that may hold memory
                for name in ("pred_curr", "jac_curr", "jac_curr_phys"):
                    if name in locals():
                        try:
                            del locals()[name]
                        except Exception:
                            pass
                gc.collect()
                if psutil:
                    proc = psutil.Process(os.getpid())
                    print("RSS MB after GC:", proc.memory_info().rss / 1024 ** 2)
                try:
                    print(
                        f"simulate_and_grad failed for member {ne_count} at initial eval: {e}. Retry with smoothed fields.")
                    # latent_rh_max = apply_bounds_rh(np.log(rh_phys_max * 0.5)) # make stronger bounds
                    param_current = smooth_param_field_2d(param_current, dims, sigma=0.25, which='rh')
                    param_current = smooth_param_field_2d(param_current, dims, sigma=0.25, which='ratio')
                    pred_curr, jac_curr, jac_curr_phys = simulate_and_grad(param_current)
                except Exception as e:
                    try:
                        with open(f'inversion_results_assim_{tot_assim_index[el-1][0]}.pkl', 'rb') as f:
                            results_restart = pickle.load(f)
                        param_vec = results_restart.get("prior_params")
                        del results_restart
                        if apply_bounds:
                            param_vec[:dims[0] * dims[2], :] = apply_bounds_rh(param_vec[:dims[0] * dims[2], :])
                        print(
                            f"simulate_and_grad failed for member {ne_count} at initial eval: {e}. Retry with result from previous assim step.")
                        param_current = param_vec[:, ne_count]
                        pred_curr, jac_curr, jac_curr_phys = simulate_and_grad(param_current)
                    except Exception as e:
                        print(
                            f"simulate_and_grad failed for member {ne_count} at initial eval: {e}. Skipping this member for now.")
                        failed_indices.append(ne_count)
                        continue  # skip this ensemble member

            # compute initial loss
            curr_loss = custom_loss(pred_curr, data_current, param_current, mean_theta, c_theta, Cd_inflated)
            print(f"Initial Loss: {curr_loss}")
            step_lambda = 10 ** np.floor(np.log10(curr_loss))

            # Levenberg-Marquardt style optimization loop (unchanged)
            for iter in range(max_iterations):
                accepted = False
                used_lambda = step_lambda

                for _ in range(max_lm_trials):
                    used_lambda = step_lambda
                    search_dir = map_update_data_space_LM(
                        param_current, pred_curr, mean_theta, jac_curr, data_current, Cd_inflated, c_theta, used_lambda
                    )
                    search_dir = disallow_updates_prior_to_bit(search_dir, boundary_layer=no_fixed_cells_z)
                    trial_param = param_current + search_dir

                    try:
                        pred_trial, jac_trial, jac_trial_phys = simulate_and_grad(trial_param)
                        trial_loss = custom_loss(pred_trial, data_current, trial_param, mean_theta, c_theta,
                                                 Cd_inflated)
                    except Exception as e:
                        print(
                            f"simulate_and_grad failed during trial for member {ne_count}: {e}. Treating trial as rejected.")
                        trial_loss = np.inf
                        pred_trial = None

                    if np.isfinite(trial_loss) and trial_loss < curr_loss:
                        accepted = True
                        param_current = trial_param
                        curr_loss = trial_loss
                        jac_curr = jac_trial
                        jac_curr_phys = jac_trial_phys
                        pred_curr = pred_trial
                        step_lambda = max(step_lambda / lambda_down, lambda_min)
                        break

                    step_lambda = min(step_lambda * lambda_up, lambda_max)

                tot_loss.append(curr_loss)
                if accepted:
                    print(f"Iteration {iter}, Loss: {curr_loss} (accepted, lambda={used_lambda:.3g})")
                else:
                    print(f"Iteration {iter}, Loss: {curr_loss} (LM step rejected, lambda={used_lambda:.3g})")
                    break

            # successful completion for this member — store results and remember index
            post_param.append(param_current)
            post_pred.append(pred_curr)
            post_loss.append(curr_loss)
            post_jac.append(jac_curr)
            post_jac_phys.append(jac_curr_phys)
            success_indices.append(ne_count)

        # End per-member loop
        n_failed = len(failed_indices)
        if n_failed > 0:
            print(f"{n_failed} ensemble members failed during optimization: {failed_indices}")
            if len(success_indices) == 0:
                # no successful members — fallback: fill with prior mean or prior samples
                print("No successful members — filling with prior mean parameters.")
                for _ in range(Ne):
                    post_param.append(mean_theta.copy())
                    # Optionally set placeholders for pred/jac/loss
                    post_pred.append(np.zeros_like(post_pred[0]) if post_pred else np.zeros((len(data_vec),)))
                    post_loss.append(np.inf)
                    post_jac.append(
                        np.zeros_like(post_jac[0]) if post_jac else np.zeros((len(data_vec), param_vec.shape[0])))
                    post_jac_phys.append(np.zeros_like(post_jac_phys[0]) if post_jac_phys else np.zeros(
                        (len(data_vec), param_vec.shape[0])))
            else:
                # Fill failed slots by randomly duplicating successful members (with replacement)
                import random

                for _ in failed_indices:
                    idx = random.choice(range(len(success_indices)))  # choose one successful result
                    post_param.append(post_param[idx].copy())
                    post_pred.append(post_pred[idx].copy())
                    post_loss.append(post_loss[idx])
                    post_jac.append(post_jac[idx])
                    post_jac_phys.append(post_jac_phys[idx])
            print("Filled failed ensemble members with successful replicas.")



        #assess_intermediate_param_values(2, post_estim=True)
        if not debug:
            plot_ensemble_predictions(
                np.array(post_pred).T, data_real, data_vec,
                title_suffix=f'After Gradient Iterations',
                filename=f'ensemble_posterior_predictions_assim_step{assim_index[0]}.png'
                )

        prior_rml = copy.deepcopy(pr)
        post_mda = copy.deepcopy(param_vec)
        post_rml = copy.deepcopy(post_param)
        mean_theta_prior_rml = copy.deepcopy(mean_theta)

        if apply_bounds:
            prior_rml[:dims[0] * dims[2],:] = remove_bounds_rh(prior_rml[:dims[0] * dims[2],:])
            post_mda[:dims[0] * dims[2],:] = remove_bounds_rh(post_mda[:dims[0] * dims[2],:])
            mean_theta_prior_rml[:dims[0] * dims[2]] =  remove_bounds_rh(mean_theta[:dims[0] * dims[2]])
            for ens_no, post_rml_ens in enumerate(post_rml):
                post_rml[ens_no][:dims[0] * dims[2]] = remove_bounds_rh(post_rml_ens[:dims[0] * dims[2]])

        # save results
        results_dict = {
            'prior_params': prior_rml,
            'post_mda_params': post_mda,
            'posterior_params': post_rml,
            'posterior_predictions': post_pred,
            'posterior_losses': post_loss,
            'posterior_jacobian': post_jac,
            'posterior_jacobian_phys': post_jac_phys,
            'prior_mean_rml': mean_theta_prior_rml,
            'prior_covariance_rml': c_theta
        }
        # save to pickle
        with open(f'inversion_results_assim_{assim_index[0]}.pkl', 'wb') as f:
            pickle.dump(results_dict, f)

        # free memory:
        for name in ("prior_rml", "post_mda", "post_rml", "post_rml_ens", "post_pred", "en_pred",
                         "post_jac", "post_jac_phys", "trial_param", "param_vec", "param_vec_list", "param_current"
                         "mean_theta_prior_rml", "c_theta", "prior_rml"
                         "pr", "mean_theta", "results_restart", "jac_curr","jac_curr_phys","jac_trial", "jac_trial_phys"):
            if name in locals():
                try:
                    del locals()[name]
                except Exception:
                    try:
                        globals().pop(name, None)
                    except Exception:
                        pass

        del results_dict
        gc.collect()

        if psutil:
            proc = psutil.Process(os.getpid())
            print("RSS MB after cleanup:", proc.memory_info().rss / 1024 ** 2)
