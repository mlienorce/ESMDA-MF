import argparse
import pickle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm

DIMS = (22,3,22)
MIN_RATIO = 1.0
MAX_RATIO = 5.0
METER_TO_FEET = 3.28084
CELL_THICKNESS_FT = 1.0 * METER_TO_FEET
DX = CELL_THICKNESS_FT * 1.0 * 4.0
DZ = CELL_THICKNESS_FT * 2.0 

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
    project_root = script_dir.parent
    parser = argparse.ArgumentParser(description="Plot posterior predictions and parameters from ESMDA-Hybrid run.")
    parser.add_argument("--assim-ind", type=int, required=True, help="Assimilation index used in inversion_results_assim_{assim_ind}.pkl")
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
        "--logging-ind",
        type=int,
        default=None,
        help="Logging point index in the reference model. Defaults to assim-ind.",
    )
    parser.add_argument(
        "--reference-model",
        type=Path,
        default=project_root / "inversion" / "data" / "Benchmark-3" / "globalmodel.h5",
        help="Path to reference globalmodel.h5 used to locate the logging point.",
    )
    parser.add_argument("--data-file", type=Path, default=script_dir / "data.pkl", help="Path to true data pickle.")
    parser.add_argument("--var-file", type=Path, default=script_dir / "var.pkl", help="Path to variance pickle.")
    parser.add_argument("--outdir", type=Path, default=script_dir, help="Directory where figures are written.")
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
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
    return post_param, post_pred, post_mda_param, prior_param


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


def plot_predictions(
    post_pred: np.ndarray,
    true_data: np.ndarray,
    variances: np.ndarray,
    assim_ind: int,
    outdir: Path,
) -> Path:
    n_selected = len(SELECTED_DATA)
    max_points = min(post_pred.shape[1], true_data.size, variances.size, len(TOOLS) * n_selected)
    n_tools = max_points // n_selected
    if n_tools == 0:
        raise ValueError("Not enough points to form tool responses for selected data.")

    n_points = n_tools * n_selected
    pred = post_pred[:, :n_points]
    true_vals = true_data[:n_points]
    var_vals = np.maximum(variances[:n_points], 0.0)

    # data ordering is [tool0 selected-data block, tool1 selected-data block, ...]
    pred_mat = pred.reshape(pred.shape[0], n_tools, n_selected)
    true_mat = true_vals.reshape(n_tools, n_selected)
    sigma_mat = np.sqrt(var_vals.reshape(n_tools, n_selected))

    p05 = np.percentile(pred_mat, 5, axis=0)
    p50 = np.percentile(pred_mat, 50, axis=0)
    p95 = np.percentile(pred_mat, 95, axis=0)

    n_cols = 2
    n_rows = int(np.ceil(n_selected / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.8 * n_rows), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    x = np.arange(n_tools)
    tool_labels = [f"{freq}\n{dist}" for freq, dist in TOOLS[:n_tools]]

    for idx, data_name in enumerate(SELECTED_DATA):
        ax = axes[idx]
        ax.fill_between(x, p05[:, idx], p95[:, idx], alpha=0.25, color="tab:blue", label="Pred. p05-p95")
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

    fig.suptitle(f"Posterior Predictions Across Realizations (assim {assim_ind})")

    out_path = outdir / f"posterior_predictions_assim_{assim_ind}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_field_panel(
    ax: plt.Axes,
    field_nz_nx: np.ndarray,
    grid_axes: dict[str, np.ndarray | float],
    title: str,
    cmap: str,
):
    x_edges = np.asarray(grid_axes["x_edges"])
    z_edges = np.asarray(grid_axes["z_edges"])
    well_x_ft = float(grid_axes["well_x_ft"])
    well_tvd_ft = float(grid_axes["well_tvd_ft"])

    mesh = ax.pcolormesh(x_edges, z_edges, field_nz_nx, shading="auto", cmap=cmap)
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
    assim_ind: int,
    outdir: Path,
    grid_axes: dict[str, np.ndarray | float],
    name: str = 'post',
) -> Path:
    rh_mean = np.median(rh, axis=0)
    rv_mean = np.median(rv, axis=0)
    rh_std = rh.std(axis=0)
    rv_std = rv.std(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    im0 = _plot_field_panel(
        axes[0, 0],
        rh_mean.T,
        grid_axes,
        "Mean rh across realizations",
        "viridis",
    )
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = _plot_field_panel(
        axes[0, 1],
        rv_mean.T,
        grid_axes,
        "Mean rv across realizations",
        "viridis",
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

    out_path = outdir / f"posterior_parameters_{name}_assim_{assim_ind}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()

    data_row = args.assim_ind if args.data_row is None else args.data_row
    logging_ind = args.assim_ind if args.logging_ind is None else args.logging_ind
    results_file = (
        args.results_file
        if args.results_file is not None
        else Path(__file__).resolve().parent / f"inversion_results_assim_{args.assim_ind}.pkl"
    )

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    if not args.data_file.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    if not args.var_file.exists():
        raise FileNotFoundError(f"Variance file not found: {args.var_file}")
    if not args.reference_model.exists():
        raise FileNotFoundError(f"Reference model not found: {args.reference_model}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    grid_axes = build_local_grid_axes(args.reference_model, logging_ind)
    post_param, post_pred, post_mda_param, prior_param = load_results(results_file)
    true_data, variances = load_true_data_and_var(args.data_file, args.var_file, data_row=data_row)
    rh, rv = reconstruct_rh_rv(post_param)
    pred_plot = plot_predictions(post_pred, true_data, variances, args.assim_ind, args.outdir)
    param_plot = plot_parameters(rh, rv, args.assim_ind, args.outdir, grid_axes)

    print(f"Saved: {pred_plot}")
    print(f"Saved: {param_plot}")

    if post_mda_param is not None:
        rh_mda, rv_mda = reconstruct_rh_rv(np.asarray(post_mda_param).T)
        mda_param_plot = plot_parameters(rh_mda, rv_mda, args.assim_ind, args.outdir, grid_axes, "mda")
        print(f"Saved: {mda_param_plot}")

    if prior_param is not None:
        prior_rh, prior_rv = reconstruct_rh_rv(np.asarray(prior_param).T)
        prior_param_plot = plot_parameters(
            prior_rh,
            prior_rv,
            args.assim_ind,
            args.outdir,
            grid_axes,
            "prior",
        )
        print(f"Saved: {prior_param_plot}")


if __name__ == "__main__":
    main()
