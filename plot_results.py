import argparse
import pickle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
import imageio.v2 as imageio
from pathlib import Path
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import TwoSlopeNorm
import re

from typing import Iterable

#from run_DA import post_loss
# python plot_results.py --assim-ind 30 --assim-range 25 35 --outdir Plotting --rh-clim 1 30 --rv-clim 1 30 --aspect-to-plot Mean"
DIMS = (23,3,34)
#DIMS = (19,3,20)
MIN_RATIO = 1.0
MAX_RATIO = 4.0
METER_TO_FEET = 3.28084
CELL_THICKNESS_FT = 1.0 * METER_TO_FEET
DX = 5#CELL_THICKNESS_FT * 1.0 * 4.0
DZ = 0.437*12#CELL_THICKNESS_FT * 1.0


TOOLS = [
    ("6kHz", "83ft"),
    ("12kHz", "83ft"),
    ("24kHz", "83ft"),
    ("24kHz", "43ft"),
    ("48kHz", "43ft"),
    ("96kHz", "43ft"),
]

OBSERVED_DATA_ORDER_BFIELD = [
    "real(Bxx)",
    "real(Bxy)",
    "real(Bxz)",
    "real(Byx)",
    "real(Byy)",
    "real(Byz)",
    "real(Bzx)",
    "real(Bzy)",
    "real(Bzz)",
    "img(Bxx)",
    "img(Bxy)",
    "img(Bxz)",
    "img(Byx)",
    "img(Byy)",
    "img(Byz)",
    "img(Bzx)",
    "img(Bzy)",
    "img(Bzz)",
]

SELECTED_DATA = [
    "real(Bxx)",
    "real(Bxz)",
    "real(Bzx)",
    "real(Bzz)",
    "img(Bxx)",
    "img(Bxz)",
    "img(Bzx)",
    "img(Bzz)",
]
SELECTED_DATA_INDICES = [OBSERVED_DATA_ORDER_BFIELD.index(name) for name in SELECTED_DATA]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = Path("/home/AD.NORCERESEARCH.NO/mlie/3DGiG/")#script_dir.parent


    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--example-folder",
        type = str,
        default = "",
        help="Name of folder containing results and data for example run (to be appended script-dir)",
    )

    # parse just this known arg (leave the rest for the full parser)
    known, _ = parser.parse_known_args()
    if known.example_folder:
        script_dir = (script_dir / known.example_folder).resolve()

        # now create the real parser (with help) and add all args using updated script_dir
        parser = argparse.ArgumentParser(description="Plot posterior predictions and parameters from ESMDA-Hybrid run.")
        parser.add_argument("--example-folder", type=str, default=known.example_folder,
                            help="Name of a subfolder under the script directory (appended to script_dir).")
        parser.add_argument("--assim-ind", type=int, required=True,
                            help="Assimilation index used in inversion_results_assim_{assim_ind}.pkl")


    parser.add_argument(
        "--data-row",
        type=int,
        default=None,
        help="Row index for data.pkl/var.pkl. Defaults to assim-ind if not provided.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Optional explicit path to inversion results pickle.",
    )

    parser.add_argument(
        "--assim-range",
        nargs=2,
        type=int,
        default=None,
        help="Optional range of assimilation indices [start end] for evolution plots.",
    )

    parser.add_argument(
        "--logging-ind",
        type=int,
        default=None,
        help="Logging point index in the reference model. Defaults to assim-ind.",
    )
    parser.add_argument(
        "--reference-model",
        type=Path,
        #default=project_root / "inversion" / "data" / "Benchmark-3" / "globalmodel.h5",
        default=project_root / "Benchmark-3" / "globalmodel.h5",
        help="Path to reference globalmodel.h5 used to locate the logging point.",
    )

    parser.add_argument(
        "--aspect-to-plot",
        choices=["Mean", "Median", "Best", "Worst"],
        type = str,
        default="Mean",
        help="Which ensemble aspect to plot."
    )

    parser.add_argument(
        "--rh-clim",
        nargs=2,
        type=float,
        default=None,
        help="Color limits for rh (vmin vmax), e.g. --rh-clim 1 40",
    )
    parser.add_argument(
        "--rv-clim",
        nargs=2,
        type=float,
        default=None,
        help="Color limits for rv (vmin vmax), e.g. --rv-clim -1 40",
    )
    parser.add_argument("--data-file", type=Path, default=script_dir / "data.pkl", help="Path to true data pickle.")
    parser.add_argument("--var-file", type=Path, default=script_dir / "var.pkl", help="Path to variance pickle.")
    parser.add_argument("--outdir", type=str, default="Plotting", help="Name of subdirectory where figures are written.")
    return parser.parse_args()


def _stack_vector_list(values: list[np.ndarray], field_name: str) -> np.ndarray:
    if not values:
        raise ValueError(f"No entries found in '{field_name}'.")
    stacked = np.asarray([np.asarray(v, dtype=float).reshape(-1) for v in values], dtype=float)
    if stacked.ndim != 2:
        raise ValueError(f"Unexpected shape for '{field_name}': {stacked.shape}")
    return stacked


def load_results(
    results_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    with results_file.open("rb") as f:
        results = pickle.load(f)

    if "posterior_params" not in results or "posterior_predictions" not in results:
        raise KeyError(
            f"Missing required keys in {results_file}. Found keys: {list(results.keys())}"
        )

    post_param = _stack_vector_list(results["posterior_params"], "posterior_params")
    post_pred = _stack_vector_list(results["posterior_predictions"], "posterior_predictions")

    post_mda_param = results.get("post_mda_params")
    prior_param = results.get("prior_params")
    post_loss = results.get("posterior_losses")
    prior_mean_rml = results.get("prior_mean_rml", None)
    prior_covariance_rml = results.get("prior_covariance_rml", None)
    post_jac = results.get("posterior_jacobian", None)
    post_jac_phys = results.get("posterior_jacobian_phys", None)
    return post_param, post_pred, post_mda_param, prior_param, post_loss, prior_mean_rml, prior_covariance_rml, post_jac, post_jac_phys


def _resolve_tool_key(row: pd.Series, tool: tuple[str, str]) -> object:
    if tool in row.index:
        return tool

    tool_normalized = str(tool).replace(" ", "")
    for col in row.index:
        if isinstance(col, str) and col.replace(" ", "") == tool_normalized:
            return col
    raise KeyError(f"Tool key {tool} not found in dataframe columns.")


def load_true_data_and_var(data_file: Path, var_file: Path, data_row: int) -> tuple[np.ndarray, np.ndarray]:
    data_df = pd.read_pickle(data_file)
    var_df = pd.read_pickle(var_file)

    if not (0 <= data_row < len(data_df)):
        raise IndexError(f"data_row={data_row} out of bounds for dataframe with {len(data_df)} rows.")

    data_series = data_df.iloc[data_row]
    var_series = var_df.iloc[data_row]

    true_data = []
    variances = []
    for tool in TOOLS:
        data_key = _resolve_tool_key(data_series, tool)
        var_key = _resolve_tool_key(var_series, tool)

        data_values = np.asarray(data_series[data_key], dtype=float)
        var_cell = var_series[var_key]

        if isinstance(var_cell, (list, tuple)) and len(var_cell) >= 2:
            var_values = np.asarray(var_cell[1], dtype=float)
        else:
            var_values = np.asarray(var_cell, dtype=float)

        true_data.append(data_values[SELECTED_DATA_INDICES])
        variances.append(var_values[SELECTED_DATA_INDICES])

    return np.concatenate(true_data), np.concatenate(variances)


def reconstruct_rh_rv(post_param: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nx, _, nz = DIMS
    n_param_per_type = nx * nz
    expected_size = 2 * n_param_per_type

    if post_param.shape[1] != expected_size:
        raise ValueError(
            f"Unexpected parameter size {post_param.shape[1]}. Expected {expected_size} "
            f"from dims={DIMS}."
        )

    rh = post_param[:, :n_param_per_type].reshape(-1, nx, nz, order="C")
    latent_ratio = post_param[:, n_param_per_type:].reshape(-1, nx, nz, order="C")

    u = norm.cdf(latent_ratio)
    ratio = MIN_RATIO * np.power(MAX_RATIO / MIN_RATIO, u)
    rv = np.log(np.exp(rh) * ratio)
    return np.exp(rh), np.exp(rv)


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("Need at least two center points to construct edges.")
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    first_step = centers[1] - centers[0]
    last_step = centers[-1] - centers[-2]
    edges[0] = centers[0] - 0.5 * first_step
    edges[-1] = centers[-1] + 0.5 * last_step
    return edges


def build_local_grid_axes(reference_model: Path, logging_ind: int) -> dict[str, np.ndarray | float]:
    with h5py.File(reference_model, "r") as f:
        wx = np.asarray(f["wellpath"]["X"]).reshape(-1)
        tvd = np.asarray(f["wellpath"]["Z"]).reshape(-1)

    if not (0 <= logging_ind < wx.size):
        raise IndexError(
            f"logging-ind={logging_ind} out of bounds for wellpath with {wx.size} points."
        )

    well_x_ft = float(wx[logging_ind] * METER_TO_FEET)
    well_tvd_ft = float(tvd[logging_ind] * METER_TO_FEET)

    nx, _, nz = DIMS
    x_centers = well_x_ft + (np.arange(nx) - nx // 2) * DX
    z_centers = well_tvd_ft + (np.arange(nz) - nz // 2) * DZ
    x_edges = _centers_to_edges(x_centers)
    z_edges = _centers_to_edges(z_centers)

    return {
        "well_x_ft": well_x_ft,
        "well_tvd_ft": well_tvd_ft,
        "x_centers": x_centers,
        "z_centers": z_centers,
        "x_edges": x_edges,
        "z_edges": z_edges,
    }





def plot_predictions_over_assim(
    assim_indices: list[int],
    results_dir: Path,
    data_file: Path,
    var_file: Path,
    outdir: Path,
    tool_slice: slice | None = None,
) -> Path:
    """
    Plot evolution of ensemble predictions and observations vs assimilation index,
    for the selected data components and tools.
    """

    assim_indices = sorted(assim_indices)
    n_assim = len(assim_indices)
    if n_assim == 0:
        raise ValueError("assim_indices is empty")

    # --- Load first step to infer dimensions
    first = assim_indices[0]
    results_file = results_dir / f"inversion_results_assim_{first}.pkl"
    if not results_file.exists():
        raise FileNotFoundError(results_file)
    post_param, post_pred0, _, _, _, _, _, _,_ = load_results(results_file)
    true_data0, variances0 = load_true_data_and_var(data_file, var_file, data_row=first)

    n_selected = len(SELECTED_DATA)
    max_points = min(
        post_pred0.shape[1],
        true_data0.size,
        variances0.size,
        len(TOOLS) * n_selected,
    )
    n_tools_full = max_points // n_selected
    if n_tools_full == 0:
        raise ValueError("Not enough points to form tool responses for selected data.")

    # Choose which tools to plot
    if tool_slice is None:
        tool_slice = slice(0, n_tools_full)
    tool_indices = range(*tool_slice.indices(n_tools_full))
    n_tools = len(tool_indices)
    n_points = n_tools * n_selected
    n_real = post_pred0.shape[0]

    # Preallocate arrays: (n_assim, n_tools, n_selected)
    p05 = np.empty((n_assim, n_tools, n_selected))
    p50 = np.empty((n_assim, n_tools, n_selected))
    p95 = np.empty((n_assim, n_tools, n_selected))
    true_mat_all = np.empty((n_assim, n_tools, n_selected))
    sigma_mat_all = np.empty((n_assim, n_tools, n_selected))

    # --- Fill over assimilation indices
    for i, aind in enumerate(assim_indices):
        results_file = results_dir / f"inversion_results_assim_{aind}.pkl"
        if not results_file.exists():
            raise FileNotFoundError(results_file)
        _, post_pred, _, _, _, _,_,_,_ = load_results(results_file)
        true_data, variances = load_true_data_and_var(data_file, var_file, data_row=aind)

        max_points = min(
            post_pred.shape[1],
            true_data.size,
            variances.size,
            len(TOOLS) * n_selected,
        )

        n_points = n_tools * n_selected

        idx_list = []
        for t in tool_indices:
            start = t * n_selected
            stop = start + n_selected
            idx_list.extend(range(start, stop))
        idx_arr = np.array(idx_list, dtype=int)

        pred = post_pred[:, idx_arr]
        true_vals = true_data[idx_arr]
        var_vals = np.maximum(variances[idx_arr], 0.0)

        pred_mat = pred.reshape(n_real, n_tools, n_selected)
        true_mat = true_vals.reshape(n_tools, n_selected)
        sigma_mat = np.sqrt(var_vals.reshape(n_tools, n_selected))

        p05[i] = np.percentile(pred_mat, 5, axis=0)
        p50[i] = np.percentile(pred_mat, 50, axis=0)
        p95[i] = np.percentile(pred_mat, 95, axis=0)
        true_mat_all[i] = true_mat
        sigma_mat_all[i] = sigma_mat

    # --- Plot vs assimilation index
    n_cols = 2
    n_rows = int(np.ceil(n_selected / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, 3.8 * n_rows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    x_assim = np.asarray(assim_indices)
    x = 5.0 * x_assim  # assim_ind = 40 -> 200 ft
    tool_dist = [f"{dist}" for _, dist in np.array(TOOLS)[list(tool_indices)]]
    if tool_slice is not None:
        tool_labels = [f"{freq}" for freq, _ in np.array(TOOLS)[list(tool_indices)]]
        suffix = tool_dist[0]
    else:
        tool_labels = [f"{freq}\n{dist}" for freq, dist in np.array(TOOLS)[list(tool_indices)]]
        suffix = ""


    colors = plt.cm.tab10(np.linspace(0, 1, n_tools))

    for idx, data_name in enumerate(SELECTED_DATA):
        ax = axes[idx]
        for t in range(n_tools):
            ax.fill_between(
                x, p05[:, t, idx], p95[:, t, idx],
                alpha=0.15, color=colors[t],
            )
            ax.plot(
                x, p50[:, t, idx],
                color=colors[t], lw=2,
                label=f"Pred median - {tool_labels[t]}" if idx == 0 else None,
            )
            ax.errorbar(
                x,
                true_mat_all[:, t, idx],
                yerr=sigma_mat_all[:, t, idx],
                fmt="o",
                color=colors[t],
                mfc="white",
                ms=4,
                ecolor=colors[t],
                elinewidth=0.8,
                capsize=2.0,
                label=f"True ±1σ - {tool_labels[t]}" if idx == 0 else None,
            )
        ax.set_title(data_name)
        if tool_slice is not None:
            ax.set_ylabel(f"Signal for tool distance {tool_dist[0]}")
        else:
            ax.set_ylabel("Signal")
        ax.grid(alpha=0.25)

    for ax in axes[n_selected:]:
        ax.axis("off")
    for ax in axes[max(0, n_selected - n_cols):n_selected]:
        ax.set_xlabel("Logging position, x [ft]")
    if n_selected > 0:
        axes[0].legend(loc="best", fontsize=8)

    fig.suptitle(f"Posterior predictions vs. logging positions")
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"posterior_predictions_over_assim_{assim_indices[0]}to{assim_indices[-1]}_{suffix}.png"

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def plot_predictions(
    post_pred: np.ndarray,
    true_data: np.ndarray,
    variances: np.ndarray,
    assim_ind: int,
    outdir: Path,
    tool_slice: slice | None = None,
    suffix: str = "",
) -> Path:
    n_selected = len(SELECTED_DATA)

    # Determine how many tools are in the full vector
    max_points = min(
        post_pred.shape[1],
        true_data.size,
        variances.size,
        len(TOOLS) * n_selected,
    )
    n_tools_full = max_points // n_selected
    if n_tools_full == 0:
        raise ValueError("Not enough points to form tool responses for selected data.")

    # Choose which tools to plot
    if tool_slice is None:
        tool_slice = slice(0, n_tools_full)
    tool_indices = range(*tool_slice.indices(n_tools_full))
    n_tools = len(tool_indices)

    # Restrict predictions/obs/vars to those tools
    # layout is [tool0 block, tool1 block, ...]
    # build an index for the selected tools
    idx_list = []
    for t in tool_indices:
        start = t * n_selected
        stop = start + n_selected
        idx_list.extend(range(start, stop))
    idx_arr = np.array(idx_list, dtype=int)

    pred = post_pred[:, idx_arr]
    true_vals = true_data[idx_arr]
    var_vals = np.maximum(variances[idx_arr], 0.0)

    pred_mat = pred.reshape(pred.shape[0], n_tools, n_selected)
    true_mat = true_vals.reshape(n_tools, n_selected)
    sigma_mat = np.sqrt(var_vals.reshape(n_tools, n_selected))
    print(np.min(sigma_mat))
    print(np.max(sigma_mat))
    print(np.min(true_mat))
    print(np.max(true_mat))

    p05 = np.percentile(pred_mat, 5, axis=0)
    p50 = np.percentile(pred_mat, 50, axis=0)
    p95 = np.percentile(pred_mat, 95, axis=0)

    n_cols = 2
    n_rows = int(np.ceil(n_selected / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(14, 3.8 * n_rows),
        sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    x = np.arange(n_tools)
    tool_labels = [f"{freq}\n{dist}" for (freq, dist) in np.array(TOOLS)[list(tool_indices)]]

    for idx, data_name in enumerate(SELECTED_DATA):
        ax = axes[idx]
        ax.fill_between(x, p05[:, idx], p95[:, idx],
                        alpha=0.25, color="tab:blue", label="Pred. p05–p95")
        ax.plot(x, p50[:, idx], color="tab:blue", lw=2, label="Pred. median")
        ax.errorbar(
            x,
            true_mat[:, idx],
            yerr=sigma_mat[:, idx],
            fmt="ko",
            ecolor="0.55",
            elinewidth=1.0,
            capsize=2.0,
            ms=4,
            label="True ±1σ",
        )
        ax.set_title(data_name)
        ax.set_ylabel("Signal")
        ax.set_xticks(x)
        ax.set_xticklabels(tool_labels)
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.legend(loc="best")

    for ax in axes[n_selected:]:
        ax.axis("off")

    for ax in axes[max(0, n_selected - n_cols):n_selected]:
        ax.set_xlabel("Tool setting")

    fig.suptitle(f"Posterior Predictions Across Realizations (assim {assim_ind}) {suffix}")

    out_path = outdir / f"posterior_predictions_assim_{assim_ind}{suffix}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def plot_model_uncertainty_from_post_jac(
    rh: np.ndarray,
    rv: np.ndarray,
    post_jac: list[np.ndarray] | np.ndarray,
    grid_axes: dict,
    outdir: Path,
    assim_ind: int,
    partial_deriv:bool = False,
    jacobian_member: int | None = None,  # None->use ensemble mean
    reg: float = 1e-6,
    data_noise_var: np.ndarray = 1.0,
    cmap: str = "viridis",
    save: bool = True,
) -> Path:
    """
    Compute and plot relative model uncertainty = std/mean from post_jac (ensemble list/array).
    - post_jac: list of (ndata, nparams) arrays or array shape (nens, ndata, nparams).
    - grid_axes: output of build_local_grid_axes(...)
      'drh_dm_bounded_vec','drv_dm_bounded_vec') required when which in ('rh_phys','rv_phys').
    """
    outdir.mkdir(parents=True, exist_ok=True)
    # normalize post_jac to stacked array (nens, ndata, nparams)
    if isinstance(post_jac, np.ndarray):
        if post_jac.ndim == 3:
            jac_stack = post_jac
        elif post_jac.ndim == 2:
            jac_stack = np.expand_dims(post_jac, axis=0)
        else:
            raise ValueError("post_jac ndarray must be 2D or 3D")
    elif isinstance(post_jac, list):
        if len(post_jac) == 0:
            raise ValueError("post_jac is empty")
        jac_stack = np.stack(post_jac, axis=0)
    else:
        raise TypeError("post_jac must be list or ndarray")

    nens, ndata, nparams = jac_stack.shape

    # choose Jacobian to use
    if jacobian_member is None:
        J = np.mean(jac_stack, axis=0)  # (ndata, nparams)
    else:
        J = jac_stack[jacobian_member]

    # infer grid dims
    nx = int(np.asarray(grid_axes["x_centers"]).size)
    nz = int(np.asarray(grid_axes["z_centers"]).size)
    nm = nx * nz
    if nparams != 2 * nm:
        raise ValueError(f"nparams ({nparams}) != 2*nx*nz ({2 * nm})")

    data_var = np.asarray(data_noise_var).reshape(-1)
    if data_var.size != ndata:
        raise ValueError(f"data_noise_var length {data_var.size} != ndata {ndata}")
    # avoid division by zero
    data_var_safe = np.maximum(data_var, 1e-16)
    Rinv = 1.0 / data_var_safe

    # compute J^T R^{-1} J efficiently: scale rows of J by Rinv then multiply
    JT_Rinv_J = J.T @ (Rinv[:, np.newaxis] * J)

    # regularization: treat small reg (<1e-12) as relative fraction of trace
    if 0 < reg < 1e-12:
        diag_reg = (np.trace(JT_Rinv_J) / nparams) * reg
    else:
        diag_reg = reg

    A = JT_Rinv_J + diag_reg * np.eye(nparams)

    # invert with fallback
    try:
        Cov = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Cov = np.linalg.pinv(A)

    std = np.sqrt(np.maximum(np.real(np.diag(Cov)), 0.0))  # (nparams,)
    std_m = std[:nm]
    std_z = std[nm:]

    eps = 1e-12

    # prepare mean denominators

    if partial_deriv: # in optimizer space
        var_name_1 = "log rh bounded"
        var_name_2 = "latent_ratio"
    else: # from simulator in physical space
        var_name_1 = "rh"
        var_name_2 = "rv"

    # Helper: reduce model array to a 1D mean vector of length nm in C-order.
    def to_mean_flat(vec, nm):
        a = np.asarray(vec)
        if a.size == nm:
            return a.reshape(-1)  # already flat
        # possible shapes: (nx,nz) or (nz,nx) or (nens,nx,nz) or (nens,nz,nx)
        if a.ndim == 3:
            # average over ensemble axis
            a = a.mean(axis=0)
        if a.ndim == 2:
            # detect orientation: prefer shape (nx,nz) used in your code, otherwise try transpose
            if a.shape == (nx, nz):
                flat = a.reshape(-1, order='C')
            elif a.shape == (nz, nx):
                flat = a.T.reshape(-1, order='C')
            else:
                raise ValueError(f"unexpected 2D shape for model array: {a.shape}")
            return flat
        raise ValueError(f"unexpected shape for model array: {a.shape}")

    denom1_vec = to_mean_flat(rh, nm)
    denom2_vec = to_mean_flat(rv, nm)

    # Determine which std vector corresponds to which denom.
    # If post_jacates are in optimizer-space, std_m/std_z map to optimizer params (m,z).
    # If post_jacates are in physical-space, caller should have passed jacobians and params
    # such that std_m corresponds to the first block (rh or similar) and std_z to the second (rv).
    rel_unc_1 = (std_m / (np.abs(denom1_vec) + eps)).reshape((nx, nz), order="C")
    rel_unc_2 = (std_z / (np.abs(denom2_vec) + eps)).reshape((nx, nz), order="C")


    field_1 = rel_unc_1.T
    field_2 = rel_unc_2.T

    vmin = min(field_1.min(), field_2.min())
    vmax = max(field_1.max(), field_2.max())

    title_1 = f"Relative uncertainty std/mean in {var_name_1} (assim {assim_ind})"
    title_2 = f"Relative uncertainty std/mean in {var_name_2} (assim {assim_ind})"
    # plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax_1 = axes[0]
    ax_2 = axes[1]
    im = _plot_field_panel(ax_1, field_1, grid_axes, title_1, cmap, vmin=vmin, vmax=vmax)
    im = _plot_field_panel(ax_2, field_2, grid_axes, title_2, cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=[ax_1, ax_2], fraction=0.046, pad=0.02)
    cb.set_label('model uncertainty')

    outpath = outdir / f"model_uncertainty_{var_name_1}_{var_name_2}_assim_{assim_ind:03d}.png"
    if save:
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        return outpath
    else:
        plt.show()
        return outdir

def plot_posterior_jacobian_assim(
    post_jac: list[np.ndarray] | np.ndarray,
    assim_ind: int,
    grid_axes: dict,
    outdir: Path,
    tool_slice: slice | None = None,
    show_svd: bool = False,
    cmap: str = "viridis",
    save: bool = True,
    partial_deriv: bool = False,
) -> Path:
    """
    Plot aggregated spatial Jacobian maps per selected data (one pair of maps per SELECTED_DATA).
    Aggregation: L2 across tool-settings for that selected data (over selected tools).
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # normalize post_jac -> (nens, ndata, nparams)
    if isinstance(post_jac, np.ndarray):
        if post_jac.ndim == 3:
            jac_stack = post_jac
        elif post_jac.ndim == 2:
            jac_stack = np.expand_dims(post_jac, axis=0)
        else:
            raise ValueError("post_jac ndarray must be 2D or 3D")
    elif isinstance(post_jac, list):
        if len(post_jac) == 0:
            raise ValueError("post_jac is empty")
        jac_stack = np.stack(post_jac, axis=0)
    else:
        raise TypeError("post_jac must be list or ndarray")

    nens, ndata, nparams = jac_stack.shape

    # grid dims
    nx = int(np.asarray(grid_axes["x_centers"]).size)
    nz = int(np.asarray(grid_axes["z_centers"]).size)
    nm = nx * nz
    if nparams != 2 * nm:
        raise ValueError(f"nparams ({nparams}) != 2*nx*nz ({2 * nm})")

    # Determine tools selection
    n_selected = len(SELECTED_DATA)
    max_points = min(n_data := ndata, len(TOOLS) * n_selected)
    n_tools_full = max_points // n_selected
    if n_tools_full == 0:
        raise ValueError("Not enough points to form tool responses for selected data.")

    if tool_slice is None:
        tool_slice = slice(0, n_tools_full)
    tool_indices = range(*tool_slice.indices(n_tools_full))

    if(tool_slice.stop - tool_slice.start) == 1:
        t_idx = list(tool_indices)[0]
        tool_entry = TOOLS[t_idx]
        if isinstance(tool_entry, (list, tuple)):
            tool_label = f"{tool_entry[0]}{tool_entry[1]}"
        else:
            tool_label = str(tool_entry)
    else:
        tool_label = f"tools{tool_slice.start}to{tool_slice.stop - 1}"

    # build index list for those tools (same ordering as predictions)
    idx_list = []
    for t in tool_indices:
        start = t * n_selected
        stop = start + n_selected
        idx_list.extend(range(start, stop))
    idx_arr = np.array(idx_list, dtype=int)
    n_tools = len(tool_indices)

    if partial_deriv: # in optimizer space
        var_name_1 = "log rh bounded"
        var_name_2 = "latent_ratio"
    else: # from simulator in physical space
        var_name_1 = "rh"
        var_name_2 = "rv"

    # ensemble-mean Jacobian (ndata, nparams)
    J_mean = np.mean(jac_stack, axis=0)

    # For each selected data index (0..n_selected-1), collect rows across chosen tools:
    # rows = [t*n_selected + data_idx for t in tool_indices]
    per_data_maps_m = []
    per_data_maps_z = []
    for data_idx in range(n_selected):
        rows = np.array([t * n_selected + data_idx for t in tool_indices], dtype=int)
        # ensure rows within bounds
        rows = rows[rows < ndata]
        if rows.size == 0:
            raise ValueError(f"No rows found for data {data_idx} with selected tools")
        # aggregate across rows: L2 across tool rows of absolute Jacobian entries, averaged over ensemble
        # First compute per-ensemble L2 across rows, then average ensemble (consistent with earlier sens)
        per_member = np.sqrt(np.sum(jac_stack[:, rows, :] ** 2, axis=1))  # (nens, nparams)
        agg = np.mean(per_member, axis=0)  # (nparams,)
        m_vec = agg[:nm].reshape((nx, nz), order="C").T  # (nz, nx)
        z_vec = agg[nm:].reshape((nx, nz), order="C").T
        per_data_maps_m.append(m_vec)
        per_data_maps_z.append(z_vec)

    # Layout: one row per selected data, 2 columns (m,z) (n_rows = ceil(n_selected/cols) but here want pairs)
    n_cols = 2
    n_rows = int(np.ceil(n_selected / 1))  # one pair per row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.8 * n_rows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    # compute global vmin/vmax for consistent color scaling
    all_vals = np.concatenate([m.ravel() for m in per_data_maps_m] + [z.ravel() for z in per_data_maps_z])
    vmin = all_vals.min()
    vmax = all_vals.max()

    for i in range(n_selected):
        row = i
        ax_m = axes[row, 0]
        ax_z = axes[row, 1]
        im_m = _plot_field_panel(ax_m, per_data_maps_m[i], grid_axes, f"{SELECTED_DATA[i]} : J ({var_name_1})", cmap, vmin=vmin,
                                 vmax=vmax)
        im_z = _plot_field_panel(ax_z, per_data_maps_z[i], grid_axes, f"{SELECTED_DATA[i]} : J ({var_name_2})", cmap, vmin=vmin,
                                 vmax=vmax)
        # only first row show legend/title handled in panel

    # turn off any extra axes if n_selected < cells
    total_cells = axes.size
    used = n_selected * n_cols
    flat_axes = axes.ravel()
    for ax in flat_axes[used:]:
        ax.axis("off")

    # shared colorbar across all panels
    fig.colorbar(im_m, ax=flat_axes[:used].tolist(), fraction=0.045, pad=0.02).set_label("|J| (aggregated)")

    fig.suptitle(f"Posterior Jacobian for {tool_label} (assim {assim_ind})", fontsize=14)

    # optional SVD saved separately
    if show_svd:
        sv = np.linalg.svd(J_mean, compute_uv=False)
        svfig, svax = plt.subplots(1, 1, figsize=(6, 3))
        svax.semilogy(np.arange(1, len(sv) + 1), sv, "-o")
        svax.set_title(f"SVD of mean J (assim {assim_ind})")
        svout = outdir / f"posterior_jacobian_assim_{assim_ind:03d}_svd.png"
        if save:
            svfig.savefig(svout, dpi=150)
            plt.close(svfig)
        else:
            plt.show()

    if partial_deriv:
        outpath = outdir / f"posterior_jacobian_assim_{assim_ind:03d}_{tool_label}_partial_deriv.png"
    else:
        outpath = outdir / f"posterior_jacobian_assim_{assim_ind:03d}_{tool_label}.png"

    if save:
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        return outpath
    else:
        plt.show()
        return outdir


def plot_posterior_jacobian_assim_org(
    post_jac: list[np.ndarray] | np.ndarray,
    assim_ind: int,
    grid_axes: dict,
    outdir: Path,
    show_svd: bool = False,
    cmap: str = "viridis",
    save: bool = True,
    partial_deriv:bool = False,
) -> Path:

    outdir.mkdir(parents=True, exist_ok=True)


    # stack ensemble -> (nens, ndata, nparams)
    jac_stack = np.stack(post_jac, axis=0)
    nens, ndata, nparams = jac_stack.shape

    # infer nx,nz from grid_axes
    nx = int(np.asarray(grid_axes["x_centers"]).size)
    nz = int(np.asarray(grid_axes["z_centers"]).size)
    nm = nx * nz
    if nparams != 2 * nm:
        raise ValueError(f"nparams ({nparams}) != 2 * nx * nz ({2*nm}). Check grid_axes or posterior_jacobian shape.")

    # L2 sensitivity per parameter for each member, then ensemble mean
    sens_per_member = np.sqrt(np.sum(jac_stack**2, axis=1))  # (nens, nparams)
    sens_mean = np.mean(sens_per_member, axis=0)            # (nparams,)

    # split m and z, reshape to (nz, nx) for _plot_field_panel which expects field_nz_nx
    sens_m = sens_mean[:nm].reshape((nx, nz), order="C").T  # -> (nz, nx)
    sens_z = sens_mean[nm:].reshape((nx, nz), order="C").T  # -> (nz, nx)

    # plotting: 2x2 grid (m map, z map, optional SVD, empty or colorbars placed)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax_m = axes[0]
    ax_z = axes[1]
    # sensible clim across both maps
    vmin = min(sens_m.min(), sens_z.min())
    vmax = max(sens_m.max(), sens_z.max())

    # reuse your helper to plot field panels (expects field_nz_nx)
    if partial_deriv: # in optimizer space
        var_name_1 = "log rh bounded"
        var_name_2 = "latent_ratio"
    else: # from simulator in physical space
        var_name_1 = "rh"
        var_name_2 = "rv"
    im_m = _plot_field_panel(ax_m, sens_m, grid_axes, f"Assim {assim_ind}: sensitivity ({var_name_1})", cmap, vmin=vmin, vmax=vmax)
    im_z = _plot_field_panel(ax_z, sens_z, grid_axes, f"Assim {assim_ind}: sensitivity ({var_name_2})", cmap, vmin=vmin, vmax=vmax)

    # add colorbars using empty axis if layout reserved, else use figure colorbar
    # put a single shared colorbar in reserved bottom-right axis for consistent size
    cb = fig.colorbar(im_m, ax=[ax_m, ax_z], fraction=0.046, pad=0.02)
    cb.set_label('sensitivity')

    fig.suptitle(f"Posterior Jacobian diagnostics (assim {assim_ind})", fontsize=14)

    if  partial_deriv:
        outpath = outdir / f"posterior_jacobian_assim_{assim_ind:03d}_partial_deriv.png"
    else:
        outpath = outdir / f"posterior_jacobian_assim_{assim_ind:03d}.png"
    if save:
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        return outpath
    else:
        plt.show()
        return outdir


def _plot_field_panel(
    ax: plt.Axes,
    field_nz_nx: np.ndarray,
    grid_axes: dict[str, np.ndarray | float],
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    x_edges = np.asarray(grid_axes["x_edges"])
    z_edges = np.asarray(grid_axes["z_edges"])
    well_x_ft = float(grid_axes["well_x_ft"])
    well_tvd_ft = float(grid_axes["well_tvd_ft"])

    mesh = ax.pcolormesh(x_edges, z_edges, field_nz_nx, shading="auto", cmap=cmap,
        vmin=vmin,
        vmax=vmax)
    ax.scatter([well_x_ft], [well_tvd_ft], marker="x", color="red", s=55, linewidths=1.5)
    ax.set_title(title)
    ax.set_xlabel("X [ft]")
    ax.set_ylabel("TVD [ft]")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(z_edges[0], z_edges[-1])
    ax.invert_yaxis()

    # Show equivalent metric distance scales on secondary axes.
    sec_x = ax.secondary_xaxis(
        "top",
        functions=(lambda x_ft: x_ft / METER_TO_FEET, lambda x_m: x_m * METER_TO_FEET),
    )
    sec_x.set_xlabel("X [m]")
    sec_x.xaxis.set_major_locator(MaxNLocator(nbins=6))

    sec_y = ax.secondary_yaxis(
        "right",
        functions=(lambda z_ft: z_ft / METER_TO_FEET, lambda z_m: z_m * METER_TO_FEET),
    )
    sec_y.set_ylabel("TVD [m]")
    sec_y.yaxis.set_major_locator(MaxNLocator(nbins=6))
    return mesh


def plot_parameters(
    rh: np.ndarray,
    rv: np.ndarray,
    post_loss: np.ndarray,
    assim_ind: int,
    outdir: Path,
    grid_axes: dict[str, np.ndarray | float],
    name: str = 'post',
    aspect_to_plot: str = 'Best',
    rh_clim: tuple[float, float] | None = None,
    rv_clim: tuple[float, float] | None = None,
) -> Path:

    if aspect_to_plot == 'Median':
        rh_plot = np.median(rh, axis=0)
        rv_plot = np.median(rv, axis=0)
    elif aspect_to_plot == 'Mean':
        rh_plot = np.mean(rh, axis=0)
        rv_plot = np.mean(rv, axis=0)
    elif aspect_to_plot == 'Worst':
        worst_idx = np.argmax(post_loss)
        rh_plot = rh[worst_idx]
        rv_plot = rv[worst_idx]
    elif aspect_to_plot == 'Best':
        best_idx = np.argmin(post_loss)
        rh_plot = rh[best_idx]
        rv_plot = rv[best_idx]
        rv_plot = gaussian_filter(rv_plot, sigma=1.0)  # tune sigma
        rh_plot = gaussian_filter(rh_plot, sigma=1.0)  # tune sigma

    rh_std = rh.std(axis=0)
    rv_std = rv.std(axis=0)

    if rh_clim is not None:
        rh_vmin, rh_vmax = rh_clim
    else:
        rh_vmin, rh_vmax = None, None  # or np.min/max if you prefer

    if rv_clim is not None:
        rv_vmin, rv_vmax = rv_clim
    else:
        rv_vmin, rv_vmax = None, None

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    #print(grid_axes)
    im0 = _plot_field_panel(
        axes[0, 0],
        rh_plot.T,
        grid_axes,
        f"{aspect_to_plot} rh across realizations",
        "viridis",
    vmin=rh_vmin,
    vmax=rh_vmax,
    )
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = _plot_field_panel(
        axes[0, 1],
        rv_plot.T,
        grid_axes,
        f"{aspect_to_plot} rv across realizations",
        "viridis",
    vmin=rv_vmin,
    vmax=rv_vmax,
    )
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = _plot_field_panel(
        axes[1, 0],
        rh_std.T,
        grid_axes,
        "Std rh across realizations",
        "magma",
    )
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = _plot_field_panel(
        axes[1, 1],
        rv_std.T,
        grid_axes,
        "Std rv across realizations",
        "magma",
    )
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    out_path = outdir / f"posterior_{aspect_to_plot}_parameters_{name}_assim_{assim_ind}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def make_parameter_evolution_gif_from_results(
    assim_indices: list[int],
    results_dir: Path,
    reference_model: Path,
    outdir: Path,
    name: str = "post",
    aspect_to_plot: str = "Best",   # "Mean","Median","Best","Worst"
    frames_per_step: int = 3,      # >=1: interpolation frames between assimilation steps
    smooth_sigma: float | None = 1.0,  # smoothing applied when aspect_to_plot == "Best"
    rh_clim: tuple[float, float] | None = None,
    rv_clim: tuple[float, float] | None = None,
    cmap_rh: str = "viridis",
    cmap_rv: str = "viridis",
    show_std: bool = True,         # include std panels (2x2) if True, else 1x2
    fps: int = 6,
    temp_dir: Path | None = None,
    save_intermediate: bool = False,
) -> Path:
    """
    Build a GIF showing evolution of estimated parameter fields (rh, rv) across assimilation steps
    with a fixed global viewport that covers all logging positions. The marker of the logging point
    moves smoothly between positions. Previous logging locations are shown as faded small markers.Expects one pickle per assimilation index named:
    inversion_results_assim_{assim_index}.pkl
    Each pickle must contain 'posterior_params' and optionally 'posterior_losses' usable by reconstruct_rh_rv.

    Returns path to generated GIF (written into outdir).
    """


    outdir.mkdir(parents=True, exist_ok=True)
    if temp_dir is None:
        temp_dir = outdir / "tmp_gif_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # ---- load per-assim posterior params and losses; build local grid axes for each logging index
    summaries: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    per_logging_axes: dict[int, dict] = {}
    for aind in assim_indices:
        pfile = results_dir / f"inversion_results_assim_{aind}.pkl"
        if not pfile.exists():
            raise FileNotFoundError(pfile)
        with pfile.open("rb") as f:
            results = pickle.load(f)
        post_params = results["posterior_params"]
        post_loss = results.get("posterior_losses", None)

        P = np.asarray(post_params)
        if P.ndim == 1:
            P = P[np.newaxis, :]
        elif P.ndim == 2 and P.shape[0] > P.shape[1]:
            P = P.T

        # reconstruct_rh_rv must be present in module scope
        rh_arr, rv_arr = reconstruct_rh_rv(P)  # expect (nreal,nx,nz) or (nx,nz)
        if rh_arr.ndim == 2:
            rh_arr = rh_arr[np.newaxis, ...]
        if rv_arr.ndim == 2:
            rv_arr = rv_arr[np.newaxis, ...]

        # summarize according to aspect_to_plot
        def summarize(a: np.ndarray, loss, aspect):
            if aspect == "Mean":
                field = np.mean(a, axis=0)
            elif aspect == "Median":
                field = np.median(a, axis=0)
            elif aspect == "Worst":
                idx = 0 if loss is None else int(np.argmax(loss))
                field = a[idx]
            else:  # Best
                idx = 0 if loss is None else int(np.argmin(loss))
                field = a[idx]
            if aspect == "Best" and smooth_sigma is not None:
                field = gaussian_filter(field, sigma=smooth_sigma)
            std = a.std(axis=0)
            return field, std

        rh_field, rh_std = summarize(rh_arr, post_loss, aspect_to_plot)
        rv_field, rv_std = summarize(rv_arr, post_loss, aspect_to_plot)
        summaries.append((rh_field, rv_field, rh_std, rv_std))

        # build and store local grid axes for this logging index (assume logging index == aind)
        ga = build_local_grid_axes(reference_model, aind)
        per_logging_axes[aind] = ga

    # ---- compute global grid (centers/edges) that encompasses all local grids
    all_x_centers = np.concatenate([per_logging_axes[a]["x_centers"] for a in assim_indices])
    all_z_centers = np.concatenate([per_logging_axes[a]["z_centers"] for a in assim_indices])
    nx_local = per_logging_axes[assim_indices[0]]["x_centers"].size
    nz_local = per_logging_axes[assim_indices[0]]["z_centers"].size
    DX = per_logging_axes[assim_indices[0]]["x_centers"][1] - per_logging_axes[assim_indices[0]]["x_centers"][0]
    DZ = per_logging_axes[assim_indices[0]]["z_centers"][1] - per_logging_axes[assim_indices[0]]["z_centers"][0]

    x_min = float(np.min(all_x_centers) - 0.5 * DX)
    x_max = float(np.max(all_x_centers) + 0.5 * DX)
    z_min = float(np.min(all_z_centers) - 0.5 * DZ)
    z_max = float(np.max(all_z_centers) + 0.5 * DZ)

    gx = max(int(np.round((x_max - x_min) / DX)), nx_local)
    gz = max(int(np.round((z_max - z_min) / DZ)), nz_local)

    x_edges_global = x_min + np.arange(gx + 1) * DX
    z_edges_global = z_min + np.arange(gz + 1) * DZ
    x_centers_global = 0.5 * (x_edges_global[:-1] + x_edges_global[1:])
    z_centers_global = 0.5 * (z_edges_global[:-1] + z_edges_global[1:])

    grid_axes_global = {
        "well_x_ft": per_logging_axes[assim_indices[0]]["well_x_ft"],
        "well_tvd_ft": per_logging_axes[assim_indices[0]]["well_tvd_ft"],
        "x_centers": x_centers_global,
        "z_centers": z_centers_global,
        "x_edges": x_edges_global,
        "z_edges": z_edges_global,
    }

    # persistent global-state arrays (nz_global, nx_global) filled with NaN
    gxN = grid_axes_global["x_centers"].size
    gzN = grid_axes_global["z_centers"].size
    global_rh_state = np.full((gzN, gxN), np.nan, dtype=float)
    global_rv_state = np.full((gzN, gxN), np.nan, dtype=float)
    global_rh_std_state = np.full((gzN, gxN), np.nan, dtype=float)
    global_rv_std_state = np.full((gzN, gxN), np.nan, dtype=float)

    # ---- helper to embed local (nx_local,nz_local) into global (gz,gx)
    def embed_to_global(local_field: np.ndarray, local_axes: dict) -> np.ndarray:
        # local_field expected shape (nz_local, nx_local) for plotting helper input
        a = np.asarray(local_field)
        # ensure local is (nx_local, nz_local) before transpose insertion
        # if given (nz, nx) from earlier code, transpose to (nx,nz)
        if a.shape == (nz_local, nx_local):
            local_nx_nz = a.T
        elif a.shape == (nx_local, nz_local):
            local_nx_nz = a
        else:
            # try to reshape in C-order fallback
            local_nx_nz = a.reshape((nx_local, nz_local), order="C")
        gxN = grid_axes_global["x_centers"].size
        gzN = grid_axes_global["z_centers"].size
        global_field = np.full((gzN, gxN), np.nan, dtype=float)  # (nz_global, nx_global)

        # compute offsets
        local_x0 = float(local_axes["x_centers"][0])
        local_z0 = float(local_axes["z_centers"][0])
        gx0 = grid_axes_global["x_centers"][0]
        gz0 = grid_axes_global["z_centers"][0]
        ix_off = int(round((local_x0 - gx0) / DX))
        iz_off = int(round((local_z0 - gz0) / DZ))
        ix0 = max(0, ix_off)
        iz0 = max(0, iz_off)
        ix1 = ix0 + local_nx_nz.shape[0]
        iz1 = iz0 + local_nx_nz.shape[1]
        # trim if outside
        if ix1 > gxN:
            local_nx_nz = local_nx_nz[: local_nx_nz.shape[0] - (ix1 - gxN), :]
            ix1 = gxN
        if iz1 > gzN:
            local_nx_nz = local_nx_nz[:, : local_nx_nz.shape[1] - (iz1 - gzN)]
            iz1 = gzN
        # assign: global_field expects (nz, nx) so transpose local block
        global_field[iz0:iz1, ix0:ix1] = local_nx_nz.T
        return global_field

    # ---- prepare color limits if not provided
    all_rh = np.concatenate([s[0].ravel() for s in summaries])
    all_rv = np.concatenate([s[1].ravel() for s in summaries])
    rh_vmin, rh_vmax = (float(np.nanpercentile(all_rh, 1)), float(np.nanpercentile(all_rh, 99))) if rh_clim is None else rh_clim
    rv_vmin, rv_vmax = (float(np.nanpercentile(all_rv, 1)), float(np.nanpercentile(all_rv, 99))) if rv_clim is None else rv_clim

    # ---- interpolation helper
    def interp_fields(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        return (1.0 - t) * a + t * b

    frame_files: list[str] = []
    n_steps = len(summaries)

    # track logging positions for previous frames (for faded trace)
    logging_positions = []  # list of (well_x_ft, well_tvd_ft)

    for i in range(n_steps):
        # current and next summaries for interpolation
        for f in range(frames_per_step):
            t = 0.0 if frames_per_step == 1 else (f / frames_per_step)
            if i < n_steps - 1:
                rhA, rvA, rhsA, rvsA = summaries[i]
                rhB, rvB, rhsB, rvsB = summaries[i + 1]
                rh_frame = interp_fields(rhA, rhB, t)
                rv_frame = interp_fields(rvA, rvB, t)
                rh_std_frame = interp_fields(rhsA, rhsB, t)
                rv_std_frame = interp_fields(rvsA, rvsB, t)
                # interpolate logging point position as well
                gaA = per_logging_axes[assim_indices[i]]
                gaB = per_logging_axes[assim_indices[i + 1]]
                wxA, wzA = gaA["well_x_ft"], gaA["well_tvd_ft"]
                wxB, wzB = gaB["well_x_ft"], gaB["well_tvd_ft"]
                wx_frame = (1 - t) * wxA + t * wxB
                wz_frame = (1 - t) * wzA + t * wzB
            else:
                rh_frame, rv_frame, rh_std_frame, rv_std_frame = summaries[i]
                ga = per_logging_axes[assim_indices[i]]
                wx_frame, wz_frame = ga["well_x_ft"], ga["well_tvd_ft"]

            # embed into global fields
            local_axes = per_logging_axes[assim_indices[i]]

            local_rh_block = embed_to_global(rh_frame.T, local_axes)  # (nz_global, nx_global)
            local_rv_block = embed_to_global(rv_frame.T, local_axes)
            local_rh_std_block = embed_to_global(rh_std_frame.T, local_axes)
            local_rv_std_block = embed_to_global(rv_std_frame.T, local_axes)

            mask_rh = np.isfinite(local_rh_block)
            mask_rv = np.isfinite(local_rv_block)
            mask_rh_std = np.isfinite(local_rh_std_block)
            mask_rv_std = np.isfinite(local_rv_std_block)

            global_rh_state[mask_rh] = local_rh_block[mask_rh]
            global_rv_state[mask_rv] = local_rv_block[mask_rv]
            global_rh_std_state[mask_rh_std] = local_rh_std_block[mask_rh_std]
            global_rv_std_state[mask_rv_std] = local_rv_std_block[mask_rv_std]

            rh_global = embed_to_global(rh_frame.T, local_axes)
            rv_global = embed_to_global(rv_frame.T, local_axes)
            rh_std_global = embed_to_global(rh_std_frame.T, local_axes)
            rv_std_global = embed_to_global(rv_std_frame.T, local_axes)

            # plotting
            if show_std:
                fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
                im0 = _plot_field_panel(axes[0, 0], rh_global, grid_axes_global, f"rh (assim {assim_indices[i]})", cmap_rh, vmin=rh_vmin, vmax=rh_vmax)
                im1 = _plot_field_panel(axes[0, 1], rv_global, grid_axes_global, f"rv (assim {assim_indices[i]})", cmap_rv, vmin=rv_vmin, vmax=rv_vmax)
                im2 = _plot_field_panel(axes[1, 0], rh_std_global, grid_axes_global, "Std rh across realizations", "magma")
                im3 = _plot_field_panel(axes[1, 1], rv_std_global, grid_axes_global, "Std rv across realizations", "magma")
                im0 = _plot_field_panel(axes[0, 0], global_rh_state, grid_axes_global, f"rh (assim {assim_indices[i]})",
                                        cmap_rh, vmin=rh_vmin, vmax=rh_vmax)
                im1 = _plot_field_panel(axes[0, 1], global_rv_state, grid_axes_global, f"rv (assim {assim_indices[i]})",
                                        cmap_rv, vmin=rv_vmin, vmax=rv_vmax)
                im2 = _plot_field_panel(axes[1, 0], global_rh_std_state, grid_axes_global, "Std rh across realizations",
                                        "magma")
                im3 = _plot_field_panel(axes[1, 1], global_rv_std_state, grid_axes_global, "Std rv across realizations",
                                        "magma")

                fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
                fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
                fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
                fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
            else:
                fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
                im0 = _plot_field_panel(axes[0], rh_global, grid_axes_global, f"rh (assim {assim_indices[i]})", cmap_rh, vmin=rh_vmin, vmax=rh_vmax)
                im1 = _plot_field_panel(axes[1], rv_global, grid_axes_global, f"rv (assim {assim_indices[i]})", cmap_rv, vmin=rv_vmin, vmax=rv_vmax)
                fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            # mark previous logging positions as faint markers
            for (px, pz) in logging_positions:
                if show_std:
                    axes[0,0].scatter([px], [pz], marker='o', color='gray', s=8, alpha=0.25)
                    axes[0,1].scatter([px], [pz], marker='o', color='gray', s=8, alpha=0.25)
                else:
                    axes[0].scatter([px], [pz], marker='o', color='gray', s=8, alpha=0.25)

            # current logging point as prominent red x
            if show_std:
                axes[0,0].scatter([wx_frame], [wz_frame], marker='x', color='red', s=60, linewidths=1.8)
                axes[0,1].scatter([wx_frame], [wz_frame], marker='x', color='red', s=60, linewidths=1.8)
            else:
                axes[0].scatter([wx_frame], [wz_frame], marker='x', color='red', s=60, linewidths=1.8)

            # append current logging point to history (use integer logging location for history)
            logging_positions.append((wx_frame, wz_frame))

            # save frame
            frame_name = temp_dir / f"frame_{i:03d}_{f:03d}.png"
            fig.suptitle(f"{name} evolution: {aspect_to_plot} (assim {assim_indices[i]})", fontsize=14)
            fig.savefig(frame_name, dpi=150)
            plt.close(fig)
            frame_files.append(str(frame_name))

    # assemble GIF
    images = [imageio.imread(fp) for fp in frame_files]
    gif_path = outdir / f"{name}_assim_evolution_{assim_indices[0]:03d}_to_{assim_indices[-1]:03d}.gif"
    imageio.mimsave(gif_path, images, fps=fps)

    if not save_intermediate:
        for fp in frame_files:
            try:
                Path(fp).unlink()
            except Exception:
                pass
        try:
            temp_dir.rmdir()
        except Exception:
            pass

    return gif_path




def plot_prior_correlation_blocks(
    prior_covariance_rml: np.ndarray,
    assim_ind: int,
    outdir: Path,
    grid_axes: dict[str, np.ndarray | float],
    ref_cell: tuple[int, int] | None = None,
    name: str = "rml_prior_corr",
) -> Path:
    """
    Plot spatial correlation pattern implied by the prior covariance, for each parameter type.
    For each block, we take the correlation between a reference cell and all cells.
    """
    x_edges = np.asarray(grid_axes["x_edges"])
    z_edges = np.asarray(grid_axes["z_edges"])
    n_x = x_edges.size - 1
    n_z = z_edges.size - 1
    n_cells = n_x * n_z

    C = prior_covariance_rml
    C_rh = C[:n_cells, :n_cells]
    C_rat = C[n_cells:2*n_cells, n_cells:2*n_cells]

    # reference cell (ix, iz): default is center
    if ref_cell is None:
        ix_ref = n_x // 2
        iz_ref = n_z // 2
    else:
        ix_ref, iz_ref = ref_cell
    k_ref = ix_ref * n_z + iz_ref  # C-order: z fastest

    # correlation with reference cell: corr(i) = C(i,k) / sqrt(C(ii)*C(kk))
    def corr_field(C_block: np.ndarray) -> np.ndarray:
        diag = np.diag(C_block)
        var_ref = diag[k_ref]
        # avoid division by zero
        denom = np.sqrt(np.maximum(diag * var_ref, 1e-18))
        corr_vec = C_block[:, k_ref] / denom
        return corr_vec.reshape(n_x, n_z, order="C")

    corr_rh = corr_field(C_rh)
    corr_rat = corr_field(C_rat)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    im0 = _plot_field_panel(
        axes[0],
        corr_rh.T,
        grid_axes,
        f"Prior correlation (log Rh) w/ cell ({ix_ref},{iz_ref})",
        "coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = _plot_field_panel(
        axes[1],
        corr_rat.T,
        grid_axes,
        f"Prior correlation (latent ratio) w/ cell ({ix_ref},{iz_ref})",
        "coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"RML prior correlation fields (assim {assim_ind})")

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{name}_blocks_assim_{assim_ind}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_rml_prior_mean_and_cov(
    prior_mean_rml: np.ndarray,
    prior_covariance_rml: np.ndarray,
    assim_ind: int,
    outdir: Path,
    grid_axes: dict[str, np.ndarray | float],
    name: str = "rml_prior_and_cov",
    rh_clim: tuple[float, float] | None = None,
    rv_clim: tuple[float, float] | None = None,
    ref_cell: tuple[int, int] | None = None,
) -> Path:
    """
    Plot RML prior mean and standard deviation for Rh and Rv.
    Uses same parameterization / grid as plot_parameters.
    """
    n_x = grid_axes["x_edges"].size - 1
    n_z = grid_axes["z_edges"].size - 1
    n_cells = n_x * n_z

    C = prior_covariance_rml
    C_rh = C[:n_cells, :n_cells]
    C_rat = C[n_cells:2 * n_cells, n_cells:2 * n_cells]

    # reference cell (ix, iz): default is center
    if ref_cell is None:
        ix_ref = n_x // 2
        iz_ref = n_z // 2
    else:
        ix_ref, iz_ref = ref_cell
    k_ref = ix_ref * n_z + iz_ref  # C-order: z fastest

    # correlation with reference cell: corr(i) = C(i,k) / sqrt(C(ii)*C(kk))
    def corr_field(C_block: np.ndarray) -> np.ndarray:
        diag = np.diag(C_block)
        var_ref = diag[k_ref]
        # avoid division by zero
        denom = np.sqrt(np.maximum(diag * var_ref, 1e-18))
        corr_vec = C_block[:, k_ref] / denom
        return corr_vec.reshape(n_x, n_z, order="C")

    corr_rh = corr_field(C_rh)
    corr_rat = corr_field(C_rat)

    # Split mean vector into Rh and ratio parts
    m_rh = prior_mean_rml[:n_cells]            # log Rh (unbounded)
    m_rat = prior_mean_rml[n_cells:2*n_cells]  # latent ratio

    # Build a fake ensemble of size 1: shape (Ne, 2*n_cells) with Ne = 1
    mean_param_vec = np.concatenate([m_rh, m_rat])  # shape (2*n_cells,)
    mean_param_vec = mean_param_vec[np.newaxis, :]  # shape (1, 2*n_cells)

    # Diagonal std from covariance
    diag = np.diag(prior_covariance_rml)
    std_rh = np.sqrt(np.maximum(diag[:n_cells], 0.0))
    std_rat = np.sqrt(np.maximum(diag[n_cells:2*n_cells], 0.0))

    # Reshape to 2D
    m_rh_2d   = m_rh.reshape(n_x, n_z, order="C")
    m_rat_2d  = m_rat.reshape(n_x, n_z, order="C")
    std_rh_2d = std_rh.reshape(n_x, n_z, order="C")
    std_rat_2d= std_rat.reshape(n_x, n_z, order="C")

    # Convert mean Rh / ratio into Rh, Rv fields (reuse your mapping)
    # Here we build a fake "ensemble" of size 1 to reuse reconstruct_rh_rv
    rh_mean, rv_mean = reconstruct_rh_rv(mean_param_vec)     # expects (2*n_cells, Ne)
    rh_mean_2d = rh_mean[0]
    rv_mean_2d = rv_mean[0]

    # Color limits
    if rh_clim is not None:
        rh_vmin, rh_vmax = rh_clim
    else:
        rh_vmin, rh_vmax = None, None
    if rv_clim is not None:
        rv_vmin, rv_vmax = rv_clim
    else:
        rv_vmin, rv_vmax = None, None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    im0 = _plot_field_panel(
        axes[0, 0],
        rh_mean_2d.T,
        grid_axes,
        "RML prior mean rh",
        "viridis",
        vmin=rh_vmin,
        vmax=rh_vmax,
    )
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = _plot_field_panel(
        axes[1, 0],
        corr_rh.T,
        grid_axes,
        f"Prior correlation (rh) w/ cell ({ix_ref},{iz_ref})",
        "coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im2 = _plot_field_panel(
        axes[0, 1],
        rv_mean_2d.T,
        grid_axes,
        "RML prior mean rv",
        "viridis",
        vmin=rv_vmin,
        vmax=rv_vmax,
    )
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)


    im3 = _plot_field_panel(
        axes[1,1],
        corr_rat.T,
        grid_axes,
        f"Prior correlation (latent ratio) w/ cell ({ix_ref},{iz_ref})",
        "coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    fig.colorbar(im3, ax=axes[1,1], fraction=0.046, pad=0.04)



    fig.suptitle(f"RML prior mean and cov (assim {assim_ind})")

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{name}_mean_cov_assim_{assim_ind}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def plot_lm_single_data_updates_ensemble_org(
    jac_stack: np.ndarray,               # (nens, nd, nm)
    post_param: np.ndarray,              # (nens, nm)
    prior_mean_rml: np.ndarray,          # (nm,)
    prior_covariance_rml: np.ndarray,    # (nm,nm)
    data_obs: np.ndarray,                # (nd,) observed data for current logging
    pred_stack: np.ndarray | None,       # (nens, nd) predicted data per member (if None, must be computed externally)
    data_var: np.ndarray,                # (nd,) per-datum variances (Cd diag)
    lam: float,                          # LM lambda scalar
    grid_axes: dict,
    outdir: Path,
    tool_slice: slice | None = None,
    max_plots: int = 6,
    cmap: str = "viridis",
    plot_z_block: bool = True,
    var_name_1: str = "rh",
    var_name_2: str =  "rv",
) -> dict:
    """
    For each ensemble member compute and plot per-data update operators:
      CM_JT = C_theta @ J.T
      intermediate = CM_JT @ H^{-1}  (computed column-wise by solving H x = e_i)
      delta_theta_i = intermediate[:, i] * rhs[i]  where rhs = r_d - J @ r_m and r_m = (theta - theta_prior) / (1+lam)
    Inputs/assumptions:
- jac_stack shape: (nens, nd, nm)
- post_param shape: (nens, nm) matching nm == 2*nx*nz
- prior_mean_rml shape: (nm,)
- prior_covariance_rml shape: (nm,nm)
- data_obs shape: (nd,)
- pred_stack (optional) shape: (nens, nd). If None this function cannot compute r_d and will raise.
- data_var shape: (nd,) (variances)    """

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    outdir.mkdir(parents=True, exist_ok=True)

    jac_stack = np.asarray(jac_stack)
    post_param = np.asarray(post_param)
    prior_mean_rml = np.asarray(prior_mean_rml).reshape(-1)
    C_theta = np.asarray(prior_covariance_rml)
    data_var = np.asarray(data_var).reshape(-1)
    if data_var.size == 0:
        raise ValueError("data_var must be provided (per-datum variances).")
    nens, nd, nm = jac_stack.shape
    if post_param.shape[0] != nens:
        raise ValueError("post_param first dim must equal nens")

    # determine data indices
    n_selected = len(SELECTED_DATA) # number of data points per Tool setting
    max_points = min(
        pred_stack.shape[1],
        len(TOOLS) * n_selected,
    )
    n_tools_full = max_points // n_selected
    if n_tools_full == 0:
        raise ValueError("Not enough points to form tool responses for selected data.")

    if tool_slice is None:
        tool_slice = slice(0, n_tools_full)
    tool_indices = range(*tool_slice.indices(n_tools_full))
    n_tools = len(tool_indices)
    n_points = n_tools * n_selected
    n_real = pred_stack.shape[0]

    # grid dims and checks
    nx = int(np.asarray(grid_axes["x_centers"]).size)
    nz = int(np.asarray(grid_axes["z_centers"]).size)
    n_param_per_type = nx * nz
    expected_nm = 2 * n_param_per_type
    if nm != expected_nm:
        raise ValueError(f"nparams nm ({nm}) != expected 2*nx*nz ({expected_nm}).")

    def vec_to_fields(vec):
        v = vec.reshape(-1)
        m = v[: nm//2].reshape((nx, nz), order="C").T
        z = v[nm//2 :].reshape((nx, nz), order="C").T
        return m, z

    results = {"members": []}

    tool_dist = [f"{dist}" for _, dist in np.array(TOOLS)[list(tool_indices)]]
    if tool_slice is not None:
        tool_label = [f"{freq}" for freq, _ in np.array(TOOLS)[list(tool_indices)]]
        suffix = tool_dist[0]
    else:
        tool_label = [f"{freq}\n{dist}" for freq, dist in np.array(TOOLS)[list(tool_indices)]]
        suffix = ""


    for ens in range(nens//nens):
        J = jac_stack[ens]            # (nd, nm)
        theta = post_param[ens].reshape(-1)  # (nm,)
        if pred_stack is None:
            raise ValueError("pred_stack required to compute r_d (observed - predicted).")
        pred = np.asarray(pred_stack)[ens].reshape(-1)  # (nd,)
        # compute r_d (data residual, observed - predicted)
        r_d = data_obs.reshape(-1) - pred

        damp_scale = 1.0 + lam
        r_m = (theta - prior_mean_rml) / damp_scale

        # rhs = r_d - J @ r_m
        rhs = r_d - (J @ r_m)

        # Build H = (1+lam)*diag(data_var) + J C_theta J^T
        J_Ct = J @ C_theta    # (nd, nm)
        H = (1.0 + lam) * np.diag(data_var) + (J_Ct @ J.T)  # (nd, nd)

        # factor H (Cholesky if possible)
        use_cholesky = False
        try:
            L = np.linalg.cholesky(H)
            use_cholesky = True
        except np.linalg.LinAlgError:
            use_cholesky = False

        def solve_H(b):
            if use_cholesky:
                y = np.linalg.solve(L, b)
                x = np.linalg.solve(L.T, y)
                return x
            else:
                return np.linalg.solve(H, b)

        CM_JT = C_theta @ J.T  # (nm, nd)

        # compute and plot for selected data indices
        member_results = {"ens": ens, "data_indices": {}, "CM_JT_sample": None}

        idx_list = []
        for t in tool_indices:
            start = t * n_selected
            stop = start + n_selected
            idx_list.extend(range(start, stop))
        idx_arr = np.array(idx_list, dtype=int)
        fig, axes = plt.subplots(n_selected, 3, figsize=(15, 25), constrained_layout=True)

        if plot_z_block:
            fig_z, axes_z = plt.subplots(n_selected, 3, figsize=(15, 25), constrained_layout=True)

        axs_idx = 0
        #for idx, data_name in enumerate(SELECTED_DATA):
        for di in idx_list:
            if not (0 <= di < nd):
                continue
            e = np.zeros(nd, dtype=float)
            e[di] = 1.0
            try:
                Hinv_col = solve_H(e)
            except Exception:
                # fallback to pseudo-inverse column
                Hinv_col = np.linalg.pinv(H)[:, di]

            intermediate_col = CM_JT @ Hinv_col   # (nm,)
            cmjt_col = CM_JT[:, di]
            delta_theta_col = intermediate_col * float(rhs[di])

            m_cm, z_cm = vec_to_fields(cmjt_col)
            m_inter, z_inter = vec_to_fields(intermediate_col)
            m_upd, z_upd = vec_to_fields(delta_theta_col)

            # save numeric for debugging
            member_results["data_indices"][di] = {
                "cmjt_col": cmjt_col,
                "intermediate_col": intermediate_col,
                "delta_theta_col": delta_theta_col,
                "rhs_val": float(rhs[di]),
            }

            # plotting: m-block
            im0 = _plot_field_panel(axes[axs_idx,0], m_cm, grid_axes, f"CM_JT;  {SELECTED_DATA[axs_idx]}", cmap, vmin=None, vmax=None)
            fig.colorbar(im0, ax=axes[axs_idx,0], fraction=0.045)
            im1 = _plot_field_panel(axes[axs_idx,1], m_inter, grid_axes, f"CM_JT @ Hinv_col;  {SELECTED_DATA[axs_idx]}", cmap, vmin=None, vmax=None)
            fig.colorbar(im1, ax=axes[axs_idx,1], fraction=0.045)
            im2 = _plot_field_panel(axes[axs_idx,2], m_upd, grid_axes, f"Parameter update;  {SELECTED_DATA[axs_idx]}", "RdBu_r", vmin=None, vmax=None)
            fig.colorbar(im2, ax=axes[axs_idx,2], fraction=0.045)


            if plot_z_block:
                im0 = _plot_field_panel(axes_z[axs_idx, 0], z_cm, grid_axes, f"CM_JT;  {SELECTED_DATA[axs_idx]}", cmap, vmin=None, vmax=None)
                fig_z.colorbar(im0, ax=axes_z[axs_idx,0], fraction=0.045)
                im1 = _plot_field_panel(axes_z[axs_idx,1], z_inter, grid_axes, f"CM_JT @ Hinv_col;  {SELECTED_DATA[axs_idx]}", cmap, vmin=None, vmax=None)
                fig_z.colorbar(im1, ax=axes_z[axs_idx,1], fraction=0.045)
                im2 = _plot_field_panel(axes_z[axs_idx,2], z_upd, grid_axes, f"Parameter update;  {SELECTED_DATA[axs_idx]}", "RdBu_r", vmin=None, vmax=None)
                fig_z.colorbar(im2, ax=axes_z[axs_idx,2], fraction=0.045)

            axs_idx += 1
        outp = Path(outdir) / f"lm_update_tool_{tool_label[0]}_{tool_dist[0]}_{var_name_1}.png"
        fig.suptitle(f"Parameter {var_name_1} for tools setting: {tool_label[0]} and {tool_dist[0]}")
        fig.savefig(outp, dpi=200)
        if plot_z_block:
            outp_2 = Path(outdir) / f"lm_update_tool_{tool_label[0]}_{tool_dist[0]}_{var_name_2}.png"
            fig_z.suptitle(f"Parameter {var_name_2} for tools setting: {tool_label[0]} and {tool_dist[0]}")
            fig_z.savefig(outp_2, dpi=200)


        results["members"].append(member_results)

    return results

def plot_lm_single_data_updates_ensemble(
    jac_stack: np.ndarray,               # (nens, nd, nm)
    post_param: np.ndarray,              # (nens, nm)
    prior_mean_rml: np.ndarray,          # (nm,)
    prior_covariance_rml: np.ndarray,    # (nm,nm)
    data_obs: np.ndarray,                # (nd,) observed data for current logging
    pred_stack: np.ndarray | None,       # (nens, nd) predicted data per member (if None, must be computed externally)
    data_var: np.ndarray,                # (nd,) per-datum variances (Cd diag)
    lam: float,                          # LM lambda scalar
    grid_axes: dict,
    outdir: Path,
    tool_slice: slice | None = None,
    max_plots: int = 6,
    cmap: str = "viridis",
    plot_z_block: bool = True,
    var_name_1: str = "rh",
    var_name_2: str =  "rv",
) -> dict:

    outdir.mkdir(parents=True, exist_ok=True)

    jac_stack = np.asarray(jac_stack)
    post_param = np.asarray(post_param)
    prior_mean_rml = np.asarray(prior_mean_rml).reshape(-1)
    C_theta = np.asarray(prior_covariance_rml)
    data_var = np.asarray(data_var).reshape(-1)
    if data_var.size == 0:
        raise ValueError("data_var must be provided (per-datum variances).")
    nens, nd, nm = jac_stack.shape
    if post_param.shape[0] != nens:
        raise ValueError("post_param first dim must equal nens")

    # determine data indices for this tool
    n_selected = len(SELECTED_DATA)    # 8 data types per tool
    max_points = min(
        pred_stack.shape[1],
        len(TOOLS) * n_selected,
    )
    n_tools_full = max_points // n_selected
    if n_tools_full == 0:
        raise ValueError("Not enough points to form tool responses for selected data.")

    if tool_slice is None:
        tool_slice = slice(0, n_tools_full)
    tool_indices = range(*tool_slice.indices(n_tools_full))

    # we expect one tool per call (tool_slice like slice(t, t+1))
    tool_dist = [f"{dist}" for _, dist in np.array(TOOLS)[list(tool_indices)]]
    if tool_slice is not None:
        tool_label = [f"{freq}" for freq, _ in np.array(TOOLS)[list(tool_indices)]]
        suffix = tool_dist[0]
    else:
        tool_label = [f"{freq}\n{dist}" for freq, dist in np.array(TOOLS)[list(tool_indices)]]
        suffix = ""

    # indices of the 8 selected data rows for this tool
    idx_list = []
    for t in tool_indices:
        start = t * n_selected
        stop = start + n_selected
        idx_list.extend(range(start, stop))
    idx_arr = np.array(idx_list, dtype=int)

    # grid dims and checks
    nx = int(np.asarray(grid_axes["x_centers"]).size)
    nz = int(np.asarray(grid_axes["z_centers"]).size)
    n_param_per_type = nx * nz
    expected_nm = 2 * n_param_per_type
    if nm != expected_nm:
        raise ValueError(f"nparams nm ({nm}) != expected 2*nx*nz ({expected_nm}).")

    def vec_to_fields(vec):
        v = vec.reshape(-1)
        m = v[: nm//2].reshape((nx, nz), order="C").T
        z = v[nm//2 :].reshape((nx, nz), order="C").T
        return m, z

    results = {"members": []}

    for ens in range(nens//nens):
        J = jac_stack[ens]                 # (nd, nm)
        theta = post_param[ens].reshape(-1)  # (nm,)
        if pred_stack is None:
            raise ValueError("pred_stack required to compute r_d (observed - predicted).")
        pred = np.asarray(pred_stack)[ens].reshape(-1)  # (nd,)

        # residuals
        r_d = data_obs.reshape(-1) - pred
        damp_scale = 1.0 + lam
        r_m = (theta - prior_mean_rml) / damp_scale
        rhs = r_d - (J @ r_m)            # (nd,)

        # H and its factorization
        J_Ct = J @ C_theta               # (nd, nm)
        H = (1.0 + lam) * np.diag(data_var) + (J_Ct @ J.T)  # (nd, nd)

        use_cholesky = False
        try:
            L = np.linalg.cholesky(H)
            use_cholesky = True
        except np.linalg.LinAlgError:
            use_cholesky = False

        def solve_H(b):
            if use_cholesky:
                y = np.linalg.solve(L, b)
                x = np.linalg.solve(L.T, y)
                return x
            else:
                return np.linalg.solve(H, b)

        CM_JT = C_theta @ J.T  # (nm, nd)

        member_results = {"ens": ens, "data_indices": {}, "CM_JT_sample": None}

        # --- figures now have one extra row for the combined update (index n_selected)
        fig, axes = plt.subplots(n_selected + 1, 3, figsize=(15, 28), constrained_layout=True)
        if plot_z_block:
            fig_z, axes_z = plt.subplots(n_selected + 1, 3, figsize=(15, 28), constrained_layout=True)

        # ---- per‑data rows (8 rows)
        for local_i, di in enumerate(idx_list):
            if not (0 <= di < nd):
                continue
            e = np.zeros(nd, dtype=float)
            e[di] = 1.0
            try:
                Hinv_col = solve_H(e)
            except Exception:
                Hinv_col = np.linalg.pinv(H)[:, di]

            intermediate_col = CM_JT @ Hinv_col   # (nm,)
            cmjt_col = CM_JT[:, di]
            delta_theta_col = intermediate_col * float(rhs[di])

            m_cm, z_cm       = vec_to_fields(cmjt_col)
            m_inter, z_inter = vec_to_fields(intermediate_col)
            m_upd, z_upd     = vec_to_fields(delta_theta_col)

            member_results["data_indices"][di] = {
                "cmjt_col": cmjt_col,
                "intermediate_col": intermediate_col,
                "delta_theta_col": delta_theta_col,
                "rhs_val": float(rhs[di]),
            }

            data_name = SELECTED_DATA[local_i]

            im0 = _plot_field_panel(
                axes[local_i, 0], m_cm, grid_axes,
                f"CM_JT; {data_name}", cmap
            )
            fig.colorbar(im0, ax=axes[local_i, 0], fraction=0.045)

            im1 = _plot_field_panel(
                axes[local_i, 1], m_inter, grid_axes,
                f"CM_JT @ H⁻¹ eᵢ; {data_name}", cmap
            )
            fig.colorbar(im1, ax=axes[local_i, 1], fraction=0.045)

            im2 = _plot_field_panel(
                axes[local_i, 2], m_upd, grid_axes,
                f"Param. update Δθᵢ; {data_name}", "RdBu_r"
            )
            fig.colorbar(im2, ax=axes[local_i, 2], fraction=0.045)

            if plot_z_block:
                im0z = _plot_field_panel(
                    axes_z[local_i, 0], z_cm, grid_axes,
                    f"CM_JT; {data_name}", cmap
                )
                fig_z.colorbar(im0z, ax=axes_z[local_i, 0], fraction=0.045)

                im1z = _plot_field_panel(
                    axes_z[local_i, 1], z_inter, grid_axes,
                    f"CM_JT @ H⁻¹ eᵢ; {data_name}", cmap
                )
                fig_z.colorbar(im1z, ax=axes_z[local_i, 1], fraction=0.045)

                im2z = _plot_field_panel(
                    axes_z[local_i, 2], z_upd, grid_axes,
                    f"Param. update Δθᵢ; {data_name}", "RdBu_r"
                )
                fig_z.colorbar(im2z, ax=axes_z[local_i, 2], fraction=0.045)

        # ---- combined row: update from all 8 data types together
        # Solve H x = rhs once, then Δθ_all = CM_JT @ x
        try:
            Hinv_rhs = solve_H(rhs)            # (nd,)
        except Exception:
            Hinv_rhs = np.linalg.pinv(H) @ rhs

        intermediate_all = CM_JT @ Hinv_rhs    # (nm,)
        delta_theta_all  = intermediate_all    # already includes rhs

        # For CM_JT panel, use a simple aggregate of CM_JT over the 8 indices, e.g. sum
        cmjt_sum = np.sum(CM_JT[:, idx_list], axis=1)  # (nm,)

        m_cm_all,       z_cm_all       = vec_to_fields(cmjt_sum)
        m_inter_all,    z_inter_all    = vec_to_fields(intermediate_all)
        m_upd_all,      z_upd_all      = vec_to_fields(delta_theta_all)

        combined_row = n_selected   # last row index

        im0 = _plot_field_panel(
            axes[combined_row, 0], m_cm_all, grid_axes,
            "CM_JT (sum over 8 data)", cmap
        )
        fig.colorbar(im0, ax=axes[combined_row, 0], fraction=0.045)

        im1 = _plot_field_panel(
            axes[combined_row, 1], m_inter_all, grid_axes,
            "CM_JT @ H⁻¹ rhs (all 8)", cmap
        )
        fig.colorbar(im1, ax=axes[combined_row, 1], fraction=0.045)

        im2 = _plot_field_panel(
            axes[combined_row, 2], m_upd_all, grid_axes,
            "Param. update Δθ (all 8)", "RdBu_r"
        )
        fig.colorbar(im2, ax=axes[combined_row, 2], fraction=0.045)

        if plot_z_block:
            im0z = _plot_field_panel(
                axes_z[combined_row, 0], z_cm_all, grid_axes,
                "CM_JT (sum over 8 data)", cmap
            )
            fig_z.colorbar(im0z, ax=axes_z[combined_row, 0], fraction=0.045)

            im1z = _plot_field_panel(
                axes_z[combined_row, 1], z_inter_all, grid_axes,
                "CM_JT @ H⁻¹ rhs (all 8)", cmap
            )
            fig_z.colorbar(im1z, ax=axes_z[combined_row, 1], fraction=0.045)

            im2z = _plot_field_panel(
                axes_z[combined_row, 2], z_upd_all, grid_axes,
                "Param. update Δθ (all 8)", "RdBu_r"
            )
            fig_z.colorbar(im2z, ax=axes_z[combined_row, 2], fraction=0.045)

        outp = Path(outdir) / f"lm_update_tool_{tool_label[0]}_{tool_dist[0]}_{var_name_1}.png"
        fig.suptitle(f"Parameter {var_name_1} for tool setting: {tool_label[0]} {tool_dist[0]}")
        fig.savefig(outp, dpi=200)

        if plot_z_block:
            outp_2 = Path(outdir) / f"lm_update_tool_{tool_label[0]}_{tool_dist[0]}_{var_name_2}.png"
            fig_z.suptitle(f"Parameter {var_name_2} for tool setting: {tool_label[0]} {tool_dist[0]}")
            fig_z.savefig(outp_2, dpi=200)

        results["members"].append(member_results)

    return results

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_lm_param_updates_all_tools(
    jac_stack,           # (nens, nd, nm) physical-space Jacobian
    post_param,          # (nens, nm)
    prior_mean_rml,      # (nm,)
    prior_covariance_rml,# (nm,nm)
    data_obs,            # (nd,)
    pred_stack,          # (nens, nd)
    data_var,            # (nd,)
    lam,                 # LM lambda
    grid_axes,
    assim_ind,
    outdir,
    var_name="rh",
    ens_idx=0,
    cmap="RdBu_r",
):
    """
    One figure: rows = 8 data types + 1 combined, columns = tools.
    Each subplot shows Δθ (parameter update) for the chosen parameter type.

    var_name: "rh" (first half of state) or "rv" (second half).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    jac_stack      = np.asarray(jac_stack)
    post_param     = np.asarray(post_param)
    prior_mean_rml = np.asarray(prior_mean_rml).reshape(-1)
    C_theta        = np.asarray(prior_covariance_rml)
    data_obs       = np.asarray(data_obs).reshape(-1)
    pred_stack     = np.asarray(pred_stack)
    data_var       = np.asarray(data_var).reshape(-1)

    nens, nd, nm = jac_stack.shape
    nx = int(np.asarray(grid_axes["x_centers"]).size)
    nz = int(np.asarray(grid_axes["z_centers"]).size)
    n_param_per_type = nx * nz
    assert nm == 2 * n_param_per_type

    n_tools    = len(TOOLS)
    n_selected = len(SELECTED_DATA)  # 8
    assert nd >= n_tools * n_selected

    # choose ensemble member to visualize
    ens = ens_idx
    J_all = jac_stack[ens]            # (nd, nm)
    theta = post_param[ens].reshape(-1)
    pred  = pred_stack[ens].reshape(-1)

    # residuals
    r_d = data_obs - pred
    damp_scale = 1.0 + lam
    r_m = (theta - prior_mean_rml) / damp_scale
    rhs = r_d - (J_all @ r_m)        # (nd,)

    # H and factorization
    J_Ct = J_all @ C_theta           # (nd, nm)
    H = (1.0 + lam) * np.diag(data_var) + (J_Ct @ J_all.T)

    try:
        L = np.linalg.cholesky(H)
        use_chol = True
    except np.linalg.LinAlgError:
        use_chol = False

    def solve_H(b):
        if use_chol:
            y = np.linalg.solve(L, b)
            return np.linalg.solve(L.T, y)
        else:
            return np.linalg.solve(H, b)

    CM_JT = C_theta @ J_all.T        # (nm, nd)

    # pick index range for the requested parameter type
    if var_name == "rh":
        start_p, stop_p = 0, n_param_per_type
    elif var_name in ("rv", "ratio"):
        start_p, stop_p = n_param_per_type, 2 * n_param_per_type
    else:
        raise ValueError("var_name must be 'rh' or 'rv'")

    def field_from_vec(vec):
        block = vec[start_p:stop_p]
        return block.reshape((nx, nz), order="C").T   # (nz, nx) for plotting

    # 9 rows (8 individual + 1 combined), columns = tools
    n_rows = n_selected + 1
    fig, axes = plt.subplots(
        n_rows, n_tools,
        figsize=(3.5 * n_tools, 2.0 * n_rows),
        sharex=True, sharey=True,
        constrained_layout=True
    )

    eps = 1e-12
    max_abs_list = []
    for t, tool in enumerate(TOOLS):
        start_idx = t * n_selected
        stop_idx = start_idx + n_selected
        idx_tool = np.arange(start_idx, stop_idx, dtype=int)

        rhs_tool = np.zeros_like(rhs)
        rhs_tool[idx_tool] = rhs[idx_tool]

        try:
            Hinv_rhs_t = solve_H(rhs_tool)
        except Exception:
            Hinv_rhs_t = np.linalg.pinv(H) @ rhs_tool

        delta_theta_all = CM_JT @ Hinv_rhs_t
        field_all = field_from_vec(delta_theta_all)
        max_abs_list.append(max(abs(float(np.nanmin(field_all))), abs(float(np.nanmax(field_all)))))

    global_max_abs = max(max_abs_list) if max_abs_list else eps
    if not np.isfinite(global_max_abs) or global_max_abs <= 0:
        global_max_abs = eps

    vmin, vmax = -global_max_abs, global_max_abs
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

    for t, tool in enumerate(TOOLS):
        # indices of the 8 selected data for this tool
        start_idx = t * n_selected
        stop_idx  = start_idx + n_selected
        idx_tool  = np.arange(start_idx, stop_idx, dtype=int)

        # ---- combined row (row = 8): all 8 data types for this tool
        # Here we zero out RHS outside this tool, so "combined from these 8 data"
        rhs_tool = np.zeros_like(rhs)
        rhs_tool[idx_tool] = rhs[idx_tool]

        try:
            Hinv_rhs_t = solve_H(rhs_tool)
        except Exception:
            Hinv_rhs_t = np.linalg.pinv(H) @ rhs_tool

        delta_theta_all = CM_JT @ Hinv_rhs_t  # (nm,)

        field_all = field_from_vec(delta_theta_all)
        eps = 1e-12  # or choose a larger floor like 1e-6 or relative (max_abs*1e-6)
        max_abs = max(abs(float(np.nanmin(field_all))), abs(float(np.nanmax(field_all))))

        # ensure non-zero positive max_abs
        if max_abs <= 0 or not np.isfinite(max_abs):
            max_abs = eps
        else:
            max_abs = max(max_abs, eps)

        #vmin, vmax = -max_abs, max_abs
        #norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

        #vmin, vmax = (float(np.min(field_all)), float(np.max(field_all)))

        # ---- per‑data (rows 0..7)
        for row_local, di in enumerate(idx_tool):
            if not (0 <= di < nd):
                continue

            e = np.zeros(nd, dtype=float)
            e[di] = 1.0

            # H^{-1} e_i
            try:
                Hinv_e = solve_H(e)
            except Exception:
                Hinv_e = np.linalg.pinv(H) @ e

            # Δθ_i = CM_JT @ H^{-1} e_i * rhs_i
            intermediate_col = CM_JT @ Hinv_e       # (nm,)
            delta_theta_i    = intermediate_col * float(rhs[di])

            field_i = field_from_vec(delta_theta_i)

            ax = axes[row_local, t]
            title = "" if t == 0 else ""
            im = _plot_field_panel(
                ax, field_i, grid_axes,
                title=title,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            im.set_norm(norm)
            # labels only on left / bottom
            if t == 0:
                ax.set_ylabel(f"{SELECTED_DATA[row_local]}")
            else:
                ax.set_ylabel("")
            if row_local < n_rows - 1:
                ax.set_xlabel("")

        row_comb = n_selected
        axc = axes[row_comb, t]
        title = "" if t == 0 else ""
        imc = _plot_field_panel(
            axc, field_all, grid_axes,
            title=title,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        imc.set_norm(norm)
        if t == 0:
            axc.set_ylabel("data types combined")
        else:
            axc.set_ylabel("")
        #axc.set_xlabel("x")

        # column titles: tool info at top row
        freq, dist = tool
        axes[0, t].set_title(f"{freq}, {dist}")
        # attach it to all axes in this column
        col_axes = list(axes[:, t])#axes[:, t]
        cbar = fig.colorbar(
            imc,
            ax=col_axes,
            orientation="horizontal",
            fraction=0.046,
            pad=0.02,
        )
        cbar.set_label(f"Δ{var_name}")
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-2, 2))  # force scientific for |x|<1e-2 or >1e2
        cbar.ax.xaxis.set_major_formatter(fmt)
        cbar.update_ticks()
        cbar.update_normal(imc)

    fig.suptitle(f"LM parameter updates Δ{var_name} per data type (rows) and tool settings (columns) at logging x-pos {assim_ind*5} [ft]")
    out_file = outdir / f"lm_updates_{var_name}_assim{assim_ind}.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)



def make_gif_from_assim_steps(
    outdir,
    a_start=0,
    a_end=0,
    var_name="rh",
    duration=1.5,   # seconds per frame
    loop=0          # 0 = loop forever
):

    gif_path = outdir / f"lm_updates_{var_name}_evolution_{a_start}_{a_end}.gif"
    print(outdir)
    # open writer once
    with imageio.get_writer(str(gif_path), mode="I", duration=duration, loop=loop) as writer:
        for assim_ind in range(a_start, a_end):
            # read the PNG you just wrote and append to GIF writer (reads one image at a time)
            png_path = outdir / f"lm_updates_{var_name}_assim{assim_ind}.png"
            img = imageio.imread(str(png_path))
            writer.append_data(img)
    print("Saved GIF:", gif_path)

def main() -> None:
    args = parse_args()

    aspect_to_plot = args.aspect_to_plot

    data_row = args.assim_ind if args.data_row is None else args.data_row
    logging_ind = args.assim_ind if args.logging_ind is None else args.logging_ind

    script_dir = Path(__file__).resolve().parent
    # if parser provided both --example-folder and --example_folder, use args.example_folder
    base_dir = (script_dir / args.example_folder).resolve() if getattr(args, "example_folder", None) else script_dir

    results_file = (
        Path(args.results_file).resolve()
        if args.results_file is not None
        else (base_dir / f"inversion_results_assim_{args.assim_ind}.pkl")
    )

    #print(args)
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    if not args.data_file.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    if not args.var_file.exists():
        raise FileNotFoundError(f"Variance file not found: {args.var_file}")
    if not args.reference_model.exists():
        raise FileNotFoundError(f"Reference model not found: {args.reference_model}")

    out_dir = (base_dir / args.outdir).resolve() if getattr(args, "outdir", None) else base_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    grid_axes = build_local_grid_axes(args.reference_model, logging_ind)
    post_param, post_pred, post_mda_param, prior_param, post_loss, prior_mean_rml, prior_covariance_rml, post_jac, post_jac_phys = load_results(results_file)
    true_data, variances = load_true_data_and_var(args.data_file, args.var_file, data_row=data_row)
    rh, rv = reconstruct_rh_rv(post_param)
    mid = len(TOOLS) // 2
    if post_jac_phys is None:  # in optimizer space
        var_name_1 = "log rh bounded"
        var_name_2 = "latent_ratio"
    else:  # from simulator in physical space
        var_name_1 = "rh"
        var_name_2 = "rv"


    # ***********make plots covering one assimilation step ***************
    plot_pred = False
    plot_param_values = True
    plot_cov = False
    plot_jac = False
    plot_param_update_step = False
    plot_intermediate_param_update_steps = False
    plot_model_uncertainty = False

    if plot_pred:
        plot_name = plot_predictions(post_pred, true_data, variances, args.assim_ind, out_dir,
        tool_slice=slice(0, mid), suffix=f"_tools_1to{mid}")
        print(f"Saved: {plot_name}")
        plot_name = plot_predictions(post_pred, true_data, variances, args.assim_ind, out_dir,
                                   tool_slice=slice(mid, len(TOOLS)), suffix=f"_tools_{mid}to{len(TOOLS)}")
        print(f"Saved: {plot_name}")

    if plot_param_values:
        param_plot = plot_parameters(rh, rv, post_loss, args.assim_ind, out_dir, grid_axes, 'post', aspect_to_plot,
                                     rh_clim=tuple(args.rh_clim) if args.rh_clim is not None else None,
                                     rv_clim=tuple(args.rv_clim) if args.rv_clim is not None else None, )
        # print(f"loss {post_loss}")
        # print(f"shape rv {np.shape(rv)}")

        print(f"Saved: {param_plot}")

        if post_mda_param is not None:
            rh_mda, rv_mda = reconstruct_rh_rv(np.asarray(post_mda_param).T)
            mda_param_plot = plot_parameters(rh_mda, rv_mda, post_loss, args.assim_ind, out_dir, grid_axes, "mda",
                                             aspect_to_plot,
                                             rh_clim=tuple(args.rh_clim) if args.rh_clim is not None else None,
                                             rv_clim=tuple(args.rv_clim) if args.rv_clim is not None else None, )
            print(f"Saved: {mda_param_plot}")

        if prior_param is not None:
            prior_rh, prior_rv = reconstruct_rh_rv(np.asarray(prior_param).T)
            prior_param_plot = plot_parameters(
                prior_rh,
                prior_rv,
                post_loss,
                args.assim_ind,
                out_dir,
                grid_axes,
                "prior", aspect_to_plot,
                rh_clim=tuple(args.rh_clim) if args.rh_clim is not None else None,
                rv_clim=tuple(args.rv_clim) if args.rv_clim is not None else None, )
            print(f"Saved: {prior_param_plot}")

        if prior_mean_rml is not None and prior_covariance_rml is not None:
            prior_rml_plot = plot_rml_prior_mean_and_cov(
                prior_mean_rml,
                prior_covariance_rml,
                args.assim_ind,
                out_dir,
                grid_axes,
                name="rml_prior",
                rh_clim=tuple(args.rh_clim) if args.rh_clim is not None else None,
                rv_clim=tuple(args.rv_clim) if args.rv_clim is not None else None,
            )
            print(f"Saved: {prior_rml_plot}")

    if prior_covariance_rml is not None and plot_cov == True:
        corr_plot = plot_prior_correlation_blocks(
            prior_covariance_rml,
            args.assim_ind,
            out_dir,
            grid_axes,
        )
        print(f"Saved: {corr_plot}")

    if post_jac is not None and plot_jac == True:
        jacobian_plot = plot_posterior_jacobian_assim_org(post_jac,
                                                          args.assim_ind, grid_axes, out_dir, partial_deriv=True)
        print(f"Saved: {jacobian_plot}")
        if post_jac_phys is not None:
            jacobian_phys_plot = plot_posterior_jacobian_assim_org(post_jac_phys,
                                                                   args.assim_ind, grid_axes, out_dir,
                                                                   partial_deriv=False)
            print(f"Saved: {jacobian_phys_plot}")
            for t in range(len(TOOLS)) and plot_jac == True:
                tool_slice = slice(t, t + 1)
                plot_name = plot_posterior_jacobian_assim(post_jac_phys, args.assim_ind,
                                                          grid_axes,
                                                          out_dir,
                                                          tool_slice=tool_slice,
                                                          show_svd=False,
                                                          partial_deriv=False,
                                                          )
                print(f"Saved: {plot_name}")

    if plot_param_update_step:
        plot_lm_param_updates_all_tools(
            jac_stack=post_jac_phys,  # (nens, nd, nm)
            post_param=post_param,  # (nens, nm)
            prior_mean_rml=prior_mean_rml,  # (nm,)
            prior_covariance_rml=prior_covariance_rml,  # (nm,nm)
            data_obs=true_data,  # (nd,) observed data for current logging
            pred_stack=post_pred,  # (nens, nd) predicted data per member (if None, must be computed externally)
            data_var=variances,  # (nd,) per-datum variances (Cd diag)
            lam=10000,  # LM lambda scalar
            grid_axes=grid_axes,
            assim_ind=args.assim_ind,
            outdir=out_dir,
            var_name="rv",
            ens_idx=0,
            cmap="RdBu_r",
        )

    if plot_intermediate_param_update_steps:
        for t in range(len(TOOLS)):
            plot_lm_single_data_updates_ensemble(
                jac_stack=post_jac_phys,  # (nens, nd, nm)
                post_param=post_param,  # (nens, nm)
                prior_mean_rml=prior_mean_rml,  # (nm,)
                prior_covariance_rml=prior_covariance_rml,  # (nm,nm)
                data_obs=true_data,  # (nd,) observed data for current logging
                pred_stack=post_pred,  # (nens, nd) predicted data per member (if None, must be computed externally)
                data_var=variances,  # (nd,) per-datum variances (Cd diag)
                lam=10000,  # LM lambda scalar
                grid_axes=grid_axes,
                outdir=out_dir,
                tool_slice=slice(t, t + 1),
                max_plots=6,
                cmap="viridis",
                plot_z_block=True,
                var_name_1=var_name_1,
                var_name_2=var_name_2,
            )

    if plot_model_uncertainty:
        plot_name = plot_model_uncertainty_from_post_jac(rh, rv,
                                                         post_jac_phys,
                                                         grid_axes,
                                                         out_dir,
                                                         args.assim_ind,
                                                         partial_deriv=False,
                                                         jacobian_member=None,  # None->use ensemble mean
                                                         data_noise_var=variances,
                                                         )
        print(f"Saved: {plot_name}")


    # ********************* evolution plots over a range of assimilation indices
    make_param_gif = False
    make_plot_predictions_over_assim = False
    make_param_update_plots = False
    make_param_update_gif = False
    #make_param_gif = True
    #make_plot_predictions_over_assim = True


    if args.assim_range is not None:
        a_start, a_end = args.assim_range
        assim_indices = list(range(a_start, a_end + 1))
        if make_param_gif:
            make_parameter_evolution_gif_from_results(assim_indices=assim_indices,
            results_dir=results_file.parent,
            reference_model= args.reference_model,
            outdir=out_dir,
            name = "post",
            aspect_to_plot = "Best",  # "Mean","Median","Best","Worst"
            frames_per_step = 1,  # >=1: number of interpolation frames between assimilation steps
            smooth_sigma = 1.0,  # smoothing applied when aspect_to_plot == "Best"
            rh_clim = tuple(args.rh_clim) if args.rh_clim is not None else None,
            rv_clim = tuple(args.rv_clim) if args.rv_clim is not None else None,
            cmap_rh = "viridis",
            cmap_rv= "viridis",
            show_std= True,  # include std panels (2x2) if True, else 1x2
            )

        if make_plot_predictions_over_assim:
            evol_plot_1 = plot_predictions_over_assim(
                assim_indices=assim_indices,
                results_dir=results_file.parent,  # or Path to where your pkl's live
                data_file=args.data_file,
                var_file=args.var_file,
                outdir=out_dir,
                tool_slice=slice(0, mid),
            )
            evol_plot_2 = plot_predictions_over_assim(
                assim_indices=assim_indices,
                results_dir=results_file.parent,  # or Path to where your pkl's live
                data_file=args.data_file,
                var_file=args.var_file,
                outdir=out_dir,
                tool_slice=slice(mid, len(TOOLS)),
            )
            print(f"Saved: {evol_plot_1} and {evol_plot_2}")

        if make_param_update_plots:
            a_start, a_end = args.assim_range
            assim_indices = list(range(a_start, a_end))
            var_name = var_name_1
            for assim_ind in assim_indices:
                results_file_tmp = base_dir / f"inversion_results_assim_{assim_ind}.pkl"
                post_param, post_pred, _, prior_param, _, prior_mean_rml, prior_covariance_rml, _, post_jac_phys = load_results(
                    results_file_tmp)
                data_row_tmp = assim_ind
                true_data, variances = load_true_data_and_var(args.data_file, args.var_file, data_row=data_row_tmp)
                plot_lm_param_updates_all_tools(
                    jac_stack=post_jac_phys,  # (nens, nd, nm)
                    post_param=post_param,  # (nens, nm)
                    prior_mean_rml=prior_mean_rml,  # (nm,)
                    prior_covariance_rml=prior_covariance_rml,  # (nm,nm)
                    data_obs=true_data,  # (nd,) observed data for current logging
                    pred_stack=post_pred,  # (nens, nd) predicted data per member (if None, must be computed externally)
                    data_var=variances,  # (nd,) per-datum variances (Cd diag)
                    lam=10000,  # LM lambda scalar
                    grid_axes= grid_axes,
                    assim_ind= assim_ind,
                    outdir=out_dir,
                    var_name= var_name,
                    ens_idx=0,
                    cmap="RdBu_r",
                )
        if make_param_update_gif:
            var_name = var_name_1
            make_gif_from_assim_steps(
                out_dir,
                a_start= a_start,
                a_end= a_end,
                var_name= var_name,
                duration=10.5,  # seconds per frame
                loop=0  # 0 = loop forever
            )



if __name__ == "__main__":
    main()
