"""
Combine ESMDA with a gradient based optimizaiton method.
1. Do N_MDA - 1 ESMDA iterations
2. Do gradient based optimization for each ensemble member for the last MDA "iteration"
"""

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

## setup the model
for folder in os.listdir('.'):
    if folder.startswith('En_') and os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)

meter_to_feet = 3.28084

dims = (22,3,22) # number of cells in each direction (x,y,z)
cell_thickness_ft = 1 * meter_to_feet  # meters to feet
dX = cell_thickness_ft * 1
dY = cell_thickness_ft * 1 * 4
dZ = cell_thickness_ft * 1 

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

min_ratio, max_ratio = 1, 5 # Bounds for the anisotropy ratio

tools = ["('6kHz','83ft')","('12kHz','83ft')","('24kHz','83ft')",
         "('24kHz','43ft')","('48kHz','43ft')","('96kHz','43ft')"]

#tools = ["('24kHz','83ft')"] #, "('96kHz','43ft')"]

#tools = ["('6kHz','83ft')","('12kHz','83ft')",
#         "('24kHz','43ft')","('48kHz','43ft')"]

#tools = ["('12kHz','83ft')","('24kHz','83ft')",
#        "('96kHz','43ft')"]

#tools = ["('96kHz','43ft')"] #

def setup_simulators(reference_model_path, data_type):
    """
    Set up the UTA0D, UTA1D, and UTA2D simulators with the given reference model path and data type.
    """
    global UTA0D_sim, UTA1D_sim, UTA2D_sim, TVD, MD, WX, WY, wellpath

    UTA_input_dict = {'toolflag': 0,
                'datasign': 3 if data_type == 'UDAR' else 0,
                'anisoflag': 1,
                'toolsetting': f'{reference_model_path}/ascii/tool.inp',
                'trajectory': f'{reference_model_path}/ascii/trajectory.DAT',
                'reference_model': f'{reference_model_path}/globalmodel.h5',
                'parallel': 1,
                'map':{'ratio': [1,3]},
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

def sample_prior_0D(ne):
    #sample the 0D prior, that is a univariate Gaussian for log-resistivity and a standard normal for the anisotropy ratio latent variable, without correlation
    log_rh_std = 1.0  # Standard deviation for log-resistivity
    rh_mean = 3.0  # Define mean resistivity value

    # Create 2x2 diagonal covariance matrix for [log_rh, ratio_latent]
    C_theta = np.diag([log_rh_std**2, 1.0])

    log_rh_samples = np.random.normal(loc=rh_mean, scale=log_rh_std, size=(ne,))
    ratio_latent_samples = np.random.normal(loc=0.0, scale=1.0, size=(ne,))
    
    # Stack samples: shape (2, ne)
    samples = np.vstack([log_rh_samples, ratio_latent_samples])
    
    mean_theta = np.array([rh_mean, 0.0])

    return samples, C_theta, mean_theta

def resample_prior_0D_to_1D(ne, param_vec_0D):
    """Resample a 0D ensemble into vertically correlated 1D columns."""
    from geostat.decomp import Cholesky

    param_vec_0D = np.asarray(param_vec_0D, dtype=float)
    if param_vec_0D.shape != (2, ne):
        raise ValueError(
            f"Expected param_vec_0D with shape (2, {ne}), got {param_vec_0D.shape}"
        )

    geostat = Cholesky()
    nz = dims[2]
    center_idx = nz // 2

    column_height_ft = nz * dZ
    var_range_m = column_height_ft / meter_to_feet
    var_range_ft = var_range_m * meter_to_feet
    var_range_cells = max(1, int(np.ceil(var_range_ft / dZ)))

    param_vec_1D = np.empty((2 * nz, ne), dtype=float)
    cov_blocks = []
    mean_blocks = []

    oneD_std = [1.7, 1.0]  # Standard deviations for the 1D fields, can be tuned

    for param_idx in range(2):
        stationary_mean = float(np.mean(param_vec_0D[param_idx]))
        stationary_std = float(oneD_std[param_idx])
        mean_vec = np.full(nz, stationary_mean, dtype=float)
        mean_blocks.append(mean_vec)

        if stationary_std <= np.finfo(float).eps:
            cov = np.eye(nz, dtype=float) * np.finfo(float).eps
            conditioned_samples = np.tile(mean_vec[:, np.newaxis], (1, ne))
            conditioned_samples[center_idx, :] = param_vec_0D[param_idx, :]
        else:
            cov = geostat.gen_cov2d(
                nz, 1, stationary_std**2, var_range_cells, 1, 0, 'exp'
            )
            unconditional_samples = geostat.gen_real(mean_vec, cov, ne).reshape(nz, ne, order='C')
            kriging_weights = cov[:, center_idx] / cov[center_idx, center_idx]
            conditioning_residual = param_vec_0D[param_idx, :] - unconditional_samples[center_idx, :]
            conditioned_samples = unconditional_samples + kriging_weights[:, np.newaxis] * conditioning_residual[np.newaxis, :]
            conditioned_samples[center_idx, :] = param_vec_0D[param_idx, :]

        start = param_idx * nz
        stop = start + nz
        param_vec_1D[start:stop, :] = conditioned_samples
        cov_blocks.append(cov)

    C_theta_1D = np.block([
        [cov_blocks[0], np.zeros_like(cov_blocks[0])],
        [np.zeros_like(cov_blocks[1]), cov_blocks[1]],
    ])
    mean_theta_1D = np.concatenate(mean_blocks)

    return param_vec_1D, C_theta_1D, mean_theta_1D

def resample_prior_1D_to_2D(ne, param_vec_1D):
    """Resample a 1D ensemble into 2D fields conditioned on the central column."""
    from geostat.decomp import Cholesky

    param_vec_1D = np.asarray(param_vec_1D, dtype=float)
    nz = dims[2]
    nx = dims[0]
    n_params_per_type = nx * nz

    if param_vec_1D.shape != (2 * nz, ne):
        raise ValueError(
            f"Expected param_vec_1D with shape ({2 * nz}, {ne}), got {param_vec_1D.shape}"
        )

    geostat = Cholesky()

    var_range_m = 5.0
    var_range_ft = var_range_m * meter_to_feet
    var_range_cells = max(1, int(np.floor(var_range_ft / dX)))
    vertical_aspect = (4 * dZ) / dX

    twoD_std = [1.7, 1.0]
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

        start = param_idx * n_params_per_type
        stop = start + n_params_per_type
        param_vec_2D[start:stop, :] = conditioned_samples
        cov_blocks.append(cov)

    C_theta_2D = np.block([
        [cov_blocks[0], np.zeros_like(cov_blocks[0])],
        [np.zeros_like(cov_blocks[1]), cov_blocks[1]],
    ])
    mean_theta_2D = np.concatenate(mean_blocks)

    return param_vec_2D, C_theta_2D, mean_theta_2D

def sample_prior(ne):
    from geostat.decomp import Cholesky
    geostat = Cholesky()

    log_rh_std = 1.7  # Standard deviation for log-resistivity
    rh_mean = 1.0  # Define mean resistivity value

    var_range =  5 * meter_to_feet # feet
    #C_log_rh = geostat.gen_cov3d(grid_size[0],grid_size[1],grid_size[2], r_std**2, np.floor(var_range/dX), 1, 1, 0, 0, 0, 'exp')
    C_log_rh = geostat.gen_cov2d(dims[0], dims[2], log_rh_std**2, np.floor(var_range/dX), (4*dZ)/dX, 0, 'exp')

    # make a block diagonal covariance matrix for two parameters
    C_theta = np.block([
        [C_log_rh, np.zeros_like(C_log_rh)],
        [np.zeros_like(C_log_rh), np.eye(C_log_rh.shape[0], dtype=C_log_rh.dtype)]
    ])

    mean_theta = np.concatenate([np.ones(np.prod([dims[0], dims[2]]))*rh_mean, np.zeros(np.prod([dims[0], dims[2]]))])

    # Sample from the multivariate normal distribution
    samples = geostat.gen_real(mean_theta, C_theta, ne)
    pr = samples.reshape(-1, samples.shape[-1], order='C')

    # testing: load previous posterior
    # with open('inversion_results_assim_0.pkl','rb') as f:
    #     pr_f = pickle.load(f)
    
    # pr_l = pr_f["posterior_params"]
    # pr = np.asarray([np.asarray(v, dtype=float).reshape(-1) for v in pr_l], dtype=float).T

    return pr, C_theta, mean_theta

def _build_param_dict(param):
    """Map optimizer parameters to simulator input dictionary."""
    n_params_per_type = dims[0] * dims[2]  # nx * nz
    # Optimizer vector uses (nx, nz) with C-order flattening (z-fastest).
    rh_2d = param[:n_params_per_type].reshape(dims[0], dims[2], order='C')
    rh_ratio_2d = param[n_params_per_type:].reshape(dims[0], dims[2], order='C')
    
    # Broadcast 2D arrays to 3D by tiling along the y-axis
    rh = np.tile(rh_2d[:, np.newaxis, :], (1, dims[1], 1))

    # make rv - using reciprocal distribution for ratio
    rh_ratio = np.tile(rh_ratio_2d[:, np.newaxis, :], (1, dims[1], 1))
    u = norm.cdf(rh_ratio)  # Map to [0, 1]
    ratio = min_ratio * np.power(max_ratio / min_ratio, u)  # Reciprocal distribution
    rv = np.log(np.exp(rh) * ratio)
    

    param_dict = {'rh': rh.flatten(order='F'),
                  'rv': rv.flatten(order='F')
        }
    return param_dict, rh_2d, rh_ratio_2d

def _build_param_dict_1D(param):
    """Map a z-column parameter vector to the 1D simulator input dictionary."""
    n_params_per_type = dims[2]
    if np.size(param) != 2 * n_params_per_type:
        raise ValueError(
            f"Expected a 1D state vector of length {2 * n_params_per_type}, got {np.size(param)}"
        )
    rh_1d = np.asarray(param[:n_params_per_type], dtype=float)
    rh_ratio_1d = np.asarray(param[n_params_per_type:], dtype=float)

    u = norm.cdf(rh_ratio_1d)
    ratio = min_ratio * np.power(max_ratio / min_ratio, u)
    rv_1d = np.log(np.exp(rh_1d) * ratio)

    param_dict = {'rh': rh_1d,
                  'rv': rv_1d
        }
    return param_dict, rh_1d, rh_ratio_1d

def _get_1d_surface_depth(logging_point_idx):
    """Return a column whose center cell is aligned with the current TVD."""
    column_top_ft = TVD[logging_point_idx] - 0.5 * dims[2] * dZ
    return [column_top_ft + i * dZ for i in range(dims[2])]

def _set_logging_point_state(logging_point_idx):
    """Update tool/grid state for current logging point."""
    UTA0D_sim.tool['tvd'] = TVD[logging_point_idx:logging_point_idx+1]
    UTA0D_sim.tool['MD'] = MD[logging_point_idx:logging_point_idx+1]
    UTA0D_sim.tool['X'] = WX[logging_point_idx:logging_point_idx+1]
    UTA0D_sim.tool['ijk'] = WY[logging_point_idx:logging_point_idx+1]
    UTA0D_sim.tool['nlog'] = 1

    UTA1D_sim.tool['tvd'] = TVD[logging_point_idx:logging_point_idx+1]
    UTA1D_sim.tool['MD'] = MD[logging_point_idx:logging_point_idx+1]
    UTA1D_sim.tool['X'] = WX[logging_point_idx:logging_point_idx+1]
    UTA1D_sim.tool['ijk'] = WY[logging_point_idx:logging_point_idx+1]
    UTA1D_sim.tool['surface_depth'] = _get_1d_surface_depth(logging_point_idx)

    UTA2D_sim.tool['tvd'] = TVD[logging_point_idx:logging_point_idx+1]
    well_x_m = wellpath['X'][logging_point_idx][0]
    well_y_m = wellpath['Y'][logging_point_idx][0]
    well_z_m = wellpath['Z'][logging_point_idx][0]
    UTA2D_sim.model['shift'] = {'x': well_x_m * meter_to_feet - dX * (dims[0]//2),
                                'y': well_y_m * meter_to_feet - dY * (dims[1]//2),
                                'z': well_z_m * meter_to_feet - dZ * (dims[2]//2)}
    UTA2D_sim.tool['MD'] = MD[logging_point_idx:logging_point_idx+1]
    UTA2D_sim.tool['X'] = WX[logging_point_idx:logging_point_idx+1]
    UTA2D_sim.tool['ijk'] = WY[logging_point_idx:logging_point_idx+1]


def simulate_pred_only(param, logging_point_idx=0, simfidelity="2D"):
    """Forward response only (no Jacobian expected)."""
    if simfidelity == "0D":
        param_dict = {'rh':param[0],
                      'rv':np.log(np.exp(param[0]) * 
                                  (min_ratio*np.power(max_ratio / min_ratio, norm.cdf(param[1]))))}
        _set_logging_point_state(logging_point_idx)
        pred = UTA0D_sim.run_fwd_sim(param_dict, 0)
        pred_vec = np.concatenate([pred[0][k][selected_data_indices_0D] for k in tools])

    elif simfidelity == "1D":
        param_dict, _, _ = _build_param_dict_1D(param)
        _set_logging_point_state(logging_point_idx)
        pred = UTA1D_sim.run_fwd_sim(param_dict, 0)
        pred_vec = np.concatenate([pred[0][k][selected_data_indices_1D] for k in tools])
        
    else:        
        param_dict, _, _ = _build_param_dict(param)
        _set_logging_point_state(logging_point_idx)
        original_jac_flag = UTA2D_sim.options.get('jacobi', True)
        UTA2D_sim.options['jacobi'] = False
        try:
            pred = UTA2D_sim.run_fwd_sim(param_dict, 0)
        finally:
            UTA2D_sim.options['jacobi'] = original_jac_flag
        pred_vec = np.concatenate([pred[0][k][selected_data_indices] for k in tools])
    return pred_vec


def simulate_and_grad(param, logging_point_idx=0):
    param_dict, rh_2d, rh_ratio_2d = _build_param_dict(param)
    
    _set_logging_point_state(logging_point_idx)
    
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
    #   z = Gaussian latent for anisotropy ratio
    #   ratio = min_ratio + (max_ratio-min_ratio)*Phi(z)
    #   Rv_2D = Rh_2D * ratio
    #
    # Needed derivatives:
    #   dRh/dm = Rh
    #   dRv/dm = Rv
    #   dRv/dz = Rh * (max_ratio-min_ratio) * phi(z)
    #
    # Therefore:
    #   d(data)/dm = d(data)/dRh * dRh/dm + d(data)/dRv * dRv/dm
    #   d(data)/dz = d(data)/dRv * dRv/dz
    # -------------------------------------------------------------------------
    j_rh = jac_list[0]  # d(data)/dRh_2D
    j_rv = jac_list[1]  # d(data)/dRv_2D

    u_2d = norm.cdf(rh_ratio_2d)
    ratio_2d = min_ratio * np.power(max_ratio / min_ratio, u_2d)  # Reciprocal distribution
    dratio_dz_2d = np.log(max_ratio / min_ratio) * ratio_2d * norm.pdf(rh_ratio_2d)

    rh_phys_2d = np.exp(rh_2d)
    rv_phys_2d = rh_phys_2d * ratio_2d

    # Keep flatten convention consistent with reshape(...).reshape(-1) above.
    rh_phys_vec = rh_phys_2d.reshape(-1, order='C')
    rv_phys_vec = rv_phys_2d.reshape(-1, order='C')
    drv_dz_vec = (rh_phys_2d * dratio_dz_2d).reshape(-1, order='C')

    j_m = j_rh * rh_phys_vec[np.newaxis, :] + j_rv * rv_phys_vec[np.newaxis, :]
    j_z = j_rv * drv_dz_vec[np.newaxis, :]

    # Jacobian wrt optimizer parameters [m, z]
    jacobian_matrix = np.hstack([j_m, j_z])

    pred_vec = np.concatenate([pred[0][k][selected_data_indices] for k in tools])

    # loss = custom_loss(pred_vec, data_real, param,Cd)
    # print(loss)

    # grad = jacobian_matrix.T @ ((data_real- pred_vec) / Cd) + np.linalg.solve(cov_res, param - mean_res.flatten())

    #hessian_approx = jacobian_matrix.T @ (jacobian_matrix / Cd[:, np.newaxis]) + inv_cm

    return pred_vec, jacobian_matrix #, hessian_approx

def custom_loss(predictions, data_real, theta, theta_mean, C_theta, Cd):
    # Example: weighted mean squared error
    residuals_data = data_real - predictions.flatten()
    data_loss = 0.5 * np.sum((residuals_data ** 2) / Cd)
    
    residuals_theta = theta.flatten() - theta_mean.flatten()
    
    theta_loss = 0.5 * residuals_theta @ np.linalg.solve(C_theta, residuals_theta)

    loss = data_loss + theta_loss
    
    return loss

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
    CM_JT = C_theta @ J.T
    H_data = damp_scale * np.diag(Cd_diag) + J @ CM_JT
    rhs = r_d - J @ r_m

    try:
        w = np.linalg.solve(H_data, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(H_data, rhs, rcond=None)[0]

    delta_theta = -r_m - CM_JT @ w
    return delta_theta



reference_model_path = '/home/AD.NORCERESEARCH.NO/krfo/CodeProjects/DISTINGUISH/Jacobian/inversion/data/Benchmark-3/globalmodel.h5'
Ne = 250
#pr, c_theta, mean_theta = sample_prior(Ne)
pr, c_theta, mean_theta = sample_prior_0D(Ne)
setup_simulators(reference_model_path='/home/AD.NORCERESEARCH.NO/krfo/CodeProjects/DISTINGUISH/Jacobian/inversion/data/Benchmark-3', data_type='Bfield')

# Load reference model for plotting
with h5py.File(reference_model_path, "r") as f:
    ref_model = h5_to_dict(f)

# initalize the parameter vector for optimization
param_vec = copy.deepcopy(pr)

tot_assim_index = [180]
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

for el,assim_index in enumerate([tot_assim_index[0]]):
    Cd_row = Cd.iloc[assim_index]
    Cd_vec = np.concatenate([np.array(Cd_row[ast.literal_eval(tool)][1])[selected_data_indices] for tool in tools])
    data_vec = np.concatenate([data.iloc[assim_index][ast.literal_eval(tool)][selected_data_indices] for tool in tools])
    threshold = len(data_vec)*2
    MDA_inflation_factor = [10.0]
    curr_loss = np.inf
    mda_counter = 0
    mda_inflation = 1/(1 - 1/MDA_inflation_factor[-1]) # to ensure a geometric series

    while curr_loss > threshold and sum([1/mda for mda in MDA_inflation_factor[:-1]]) < 0.5:
        print(f"MDA iteration {mda_counter}, inflation factor: {MDA_inflation_factor[-1]:.3f}")
        en_pred = []
        for ne in range(Ne):
            en_pred.append(simulate_pred_only(param_vec[:,ne], logging_point_idx=assim_index, simfidelity="0D"))
        
        Cd_inflated = Cd_vec * MDA_inflation_factor[-1]
        at.scale_data = np.sqrt(Cd_inflated)
        data_real = np.random.normal(loc=data_vec[:, np.newaxis], 
                                     scale=at.scale_data[:, np.newaxis], 
                                     size=(len(data_vec), Ne))
        
        pred_curr = np.array(en_pred).T  # shape (d, Ne)
        
        plot_ensemble_predictions(
        pred_curr, data_real, data_vec,
        title_suffix=f' after {mda_counter} MDA iterations',
        filename=f'ensemble_predictions_after_MDA_iter_{mda_counter}.png'
        )

        at.update(
                enX = param_vec, 
                enY = pred_curr, 
                enE = data_real
            )
        
        param_vec += at.step

        curr_loss = custom_loss(pred_curr.mean(axis=1), data_real.mean(axis=1), param_vec.mean(axis=1), mean_theta, c_theta, Cd_vec)
        print(f"Loss: {curr_loss}")

        mda_counter += 1
        MDA_inflation_factor.append(MDA_inflation_factor[-1] * mda_inflation)
        print(f"MDA condition check: {sum([1/mda for mda in MDA_inflation_factor]):.3f}")


    # Now do 1D ESMDA iterations

    pr, c_theta, mean_theta = resample_prior_0D_to_1D(Ne, param_vec) # Resample prior, conditioned on the 0D posterior

    param_vec = copy.deepcopy(pr)

    at.state_scaling = np.ones(param_vec.shape[0]) # No scaling for now, but could be tuned for better performance

    while curr_loss > threshold and sum([1/mda for mda in MDA_inflation_factor[:-1]]) < 0.8: # Half energy for 1D
        print(f"MDA iteration {mda_counter}, inflation factor: {MDA_inflation_factor[-1]:.3f}")
        en_pred = []
        for ne in range(Ne):
            en_pred.append(simulate_pred_only(param_vec[:,ne], logging_point_idx=assim_index, simfidelity="1D"))
        
        Cd_inflated = Cd_vec * MDA_inflation_factor[-1]
        at.scale_data = np.sqrt(Cd_inflated)
        data_real = np.random.normal(loc=data_vec[:, np.newaxis], 
                                     scale=at.scale_data[:, np.newaxis], 
                                     size=(len(data_vec), Ne))
        
        pred_curr = np.array(en_pred).T  # shape (d, Ne)
        
        plot_ensemble_predictions(
        pred_curr, data_real, data_vec,
        title_suffix=f' after {mda_counter} MDA iterations',
        filename=f'ensemble_predictions_after_MDA_iter_{mda_counter}.png'
        )

        at.update(
                enX = param_vec, 
                enY = pred_curr, 
                enE = data_real
            )
        
        param_vec += at.step

        curr_loss = custom_loss(pred_curr.mean(axis=1), data_real.mean(axis=1), param_vec.mean(axis=1), mean_theta, c_theta, Cd_vec)
        print(f"Loss: {curr_loss}")

        mda_counter += 1
        MDA_inflation_factor.append(MDA_inflation_factor[-1] * mda_inflation)
        print(f"MDA condition check: {sum([1/mda for mda in MDA_inflation_factor]):.3f}")


    # Finally, do 2D gradient-based optimization for each ensemble member with the last inflation factor

    # Resample the prior again, conditioned on the 1D posterior
    pr, c_theta, mean_theta = resample_prior_1D_to_2D(Ne, param_vec) # Resample prior, conditioned on the 1D posterior

    opt_NE = 10
    Ne = min(Ne, opt_NE)
    post_1D = copy.deepcopy(param_vec) # Keep the last 1D MDA ensemble for comparison
    param_vec = copy.deepcopy(pr[:, :Ne]) # Take only the first Ne samples in case resampling returns more

    final_factor = 1/(1-sum([1/mda for mda in MDA_inflation_factor[:-1]]))
    MDA_inflation_factor[-1] = final_factor # Final inflation factor to ensure proper weighting of data and prior in the last iteration
    # Optimize each ensemble member with gradient-based method
    post_param = []
    post_pred = []
    post_loss = []

    Cd_inflated = Cd_vec * MDA_inflation_factor[-1]
    data_real = np.random.normal(loc=data_vec[:, np.newaxis], 
                                    scale=np.sqrt(Cd_inflated)[:, np.newaxis], 
                                    size=(len(data_vec), Ne))

    for ne in range(Ne):
        print(f"Optimizing ensemble member {ne+1}/{Ne} with gradient-based method")

        # Ensure that the MDA conditon still applies for this member before optimization
        param_current = param_vec[:, ne]
        data_current = data_real[:, ne]
    
        tot_loss = []
        lambda_down = 10.0
        lambda_up = 5.0
        lambda_min = 1e-6
        lambda_max = 1e16
        max_lm_trials = 2
        max_iterations = 5
        # Initial evaluation
        pred_curr, jac_curr = simulate_and_grad(
            param_current, logging_point_idx=assim_index
        )
        curr_loss = custom_loss(pred_curr, data_current, param_current, mean_theta, c_theta, Cd_vec)
        print(f"Initial Loss: {curr_loss}")
        step_lambda = 10**np.floor(np.log10(curr_loss))  # Initial lambda based on loss magnitude      
        for iter in range(max_iterations):
            accepted = False
            used_lambda = step_lambda

            for _ in range(max_lm_trials):
                used_lambda = step_lambda
                search_dir = map_update_data_space_LM(
                    param_current, pred_curr, mean_theta, jac_curr, data_current, Cd_inflated, c_theta, used_lambda
                )
                trial_param = param_current + search_dir

                try:              
                    pred_trial, jac_trial = simulate_and_grad(
                        trial_param, logging_point_idx=assim_index
                    )
                    #pred_trial = simulate_pred_only(trial_param, logging_point_idx=assim_index[0])
                    trial_loss = custom_loss(pred_trial, data_current, trial_param, mean_theta, c_theta, Cd_inflated)
                except Exception:
                    trial_loss = np.inf
                    pred_trial = None

                if np.isfinite(trial_loss) and trial_loss < curr_loss:
                    accepted = True
                    param_current = trial_param
                    # Re-evaluate with Jacobian for consistency in next iteration
                    # pred_curr, jac_curr = simulate_and_grad(
                    #     param_current, data_real, Cd_inflated, logging_point_idx=assim_index[0]
                    # )
                    # curr_loss_reeval = custom_loss(pred_curr, data_real, param_current, mean_theta, c_theta, Cd_inflated)
                    
                    # # Check for simulator inconsistency
                    # if abs(curr_loss_reeval - trial_loss) > 1e-6:
                    #     print(f"  WARNING: Simulator gives different results with/without Jacobian!")
                    #     print(f"    Trial loss (no Jac): {trial_loss:.6f}")
                    #     print(f"    Re-eval loss (w/ Jac): {curr_loss_reeval:.6f}")
                    #     print(f"    Difference: {abs(curr_loss_reeval - trial_loss):.6f}")
                    
                    #curr_loss = curr_loss_reeval
                    curr_loss = trial_loss
                    jac_curr = jac_trial
                    step_lambda = max(step_lambda / lambda_down, lambda_min)
                    break

                step_lambda = min(step_lambda * lambda_up, lambda_max)

            tot_loss.append(curr_loss)
            if accepted:
                print(f"Iteration {iter}, Loss: {curr_loss} (accepted, lambda={used_lambda:.3g})")
            else:
                print(f"Iteration {iter}, Loss: {curr_loss} (LM step rejected, lambda={used_lambda:.3g})")
                break
        post_param.append(param_current)
        post_pred.append(pred_trial)
        post_loss.append(trial_loss)
    

    plot_ensemble_predictions(
        np.array(post_pred).T, data_real, data_vec,
        title_suffix=f'After Gradient Iterations',
        filename=f'ensemble_posterior_predictions.png'
        )
    # save results
    results_dict = {
        'prior_params': pr,
        'post_mda_params': param_vec,
        'posterior_params': post_param,
        'posterior_predictions': post_pred,
        'posterior_losses': post_loss
    }
    # save to pickle
    with open(f'inversion_results_assim_{tot_assim_index[0]}.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
