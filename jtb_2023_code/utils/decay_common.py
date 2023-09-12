from inferelator_velocity.times import _wrap_time
from inferelator_velocity import decay

from jtb_2023_code.figure_constants import CC_LENGTH

import gc
import anndata as ad
import numpy as np


def calc_decays(
    adata,
    velocity_key,
    output_key="decay_windows",
    output_alpha_key="output_alpha",
    include_alpha=False,
    decay_quantiles=(0.0, 0.05),
    layer="X",
    force=False,
):
    if output_key in adata.var and not force:
        return adata

    lref = adata.X if layer == "X" else adata.layers[layer]

    try:
        lref = lref.A
    except AttributeError:
        pass

    decays, decays_se, alphas = _calc_decay(
        lref,
        adata.layers[velocity_key],
        include_alpha=include_alpha,
        decay_quantiles=decay_quantiles,
    )

    adata.var[output_key] = decays
    adata.var[output_key + "_se"] = decays_se

    if include_alpha:
        adata.var[output_alpha_key] = alphas

    return adata


def calc_halflives(adata, decay_key="decay", halflife_key="halflife"):
    adata.var[halflife_key] = _halflife(adata.var[decay_key])

    return adata


def calc_decay_windows(
    adata,
    velocity_key,
    time_key,
    output_key="decay_windows",
    output_alpha_key="output_alpha",
    include_alpha=False,
    decay_quantiles=(0.0, 0.05),
    bootstrap=False,
    force=False,
    layer="X",
    t_min=0,
    t_max=80,
):
    if output_key in adata.varm and not force:
        return adata

    lref = adata.X if layer == "X" else adata.layers[layer]

    try:
        lref = lref.A
    except AttributeError:
        pass

    decays, decays_se, a, t_c = _calc_decay_windowed(
        lref,
        adata.layers[velocity_key],
        adata.obs[time_key].values,
        include_alpha=include_alpha,
        decay_quantiles=decay_quantiles,
        t_min=t_min,
        t_max=t_max,
        bootstrap=bootstrap,
    )

    adata.uns[output_key] = {
        "params": {
            "include_alpha": include_alpha,
            "decay_quantiles": list(decay_quantiles),
            "bootstrap": bootstrap,
        },
        "times": t_c,
    }

    adata.varm[output_key] = np.array(decays).T
    adata.varm[output_key + "_se"] = np.array(decays_se).T

    if include_alpha:
        adata.varm[output_alpha_key] = np.array(a).T

    return adata


def _calc_decay(
    expr,
    velo,
    include_alpha=False,
    decay_quantiles=(0.0, 0.05)
):
    return decay.calc_decay(
        expr,
        velo,
        include_alpha=include_alpha,
        decay_quantiles=decay_quantiles
    )


def _calc_decay_windowed(
    expr,
    velo,
    times,
    include_alpha=False,
    decay_quantiles=(0.0, 0.05),
    bootstrap=True,
    t_min=0,
    t_max=80,
):
    return decay.calc_decay_sliding_windows(
        expr,
        velo,
        times,
        include_alpha=include_alpha,
        decay_quantiles=decay_quantiles,
        centers=np.linspace(t_min + 0.5, t_max - 0.5, int(t_max - t_min)),
        width=1.0,
        bootstrap_estimates=bootstrap,
    )


def _halflife(decay_constants):
    hl = np.full_like(decay_constants, np.nan)
    np.divide(np.log(2), decay_constants, where=decay_constants != 0, out=hl)
    hl[np.isinf(hl)] = np.nan
    return hl


def get_decay_per_cell(data_obj, by_experiment=True):
    print("Creating decay layer")
    # Copy decay constants and velocity from the calculated data objects
    adata = ad.AnnData(
        np.full(data_obj.all_data.X.shape, np.nan, dtype=float), dtype=float
    )

    adata.var_names = data_obj.all_data.var_names
    adata.obs_names = data_obj.all_data.obs_names
    adata.var = data_obj.all_data.var.copy()

    if by_experiment:
        _iter_through = [data_obj.decay_data(*k) for k in data_obj.expts]
    else:
        _iter_through = [data_obj.all_data]

    for dd in _iter_through:
        _expt_idx = adata.obs_names.isin(dd.obs_names)

        print(f"Processing experiment ({np.sum(_expt_idx)} observations)")

        adata.X[_expt_idx, :] = decay_window_to_cell_layer(dd)

    print(f"Experiment extraction complete [GC: {gc.collect()}]")

    return adata


def decay_window_to_cell_layer(adata, programs=None, program_keys=None):
    if programs is None:
        programs = adata.var["programs"]

    if program_keys is None:
        program_keys = {
            adata.uns["programs"]["rapa_program"]: "rapamycin",
            adata.uns["programs"]["cell_cycle_program"]: "cell_cycle",
        }

    _decay_rows = np.zeros(adata.shape, dtype=np.float32)

    for p in program_keys.keys():
        _prog = program_keys[p]
        _is_rapa = program_keys[p] == "rapamycin"
        _p_idx = programs == p

        print(
            f"Extracting values for program {_prog} ({np.sum(_p_idx)} genes)"
        )

        time_windows = [
            list(x)
            for x in list(
                zip(
                    adata.uns[f"{_prog}_window_decay"]["times"] - 0.5,
                    adata.uns[f"{_prog}_window_decay"]["times"] + 0.5,
                )
            )
        ]

        if _is_rapa:
            time_windows[0][0] = None
            time_windows[-1][1] = None
        else:
            time_windows[0][0] = 0
            time_windows[-1][1] = CC_LENGTH

        _decay_rows[:, _p_idx] = _cell_decay_constants(
            adata.varm[f"{_prog}_window_decay"][_p_idx, :],
            adata.obs[f"program_{p}_time"].values,
            time_windows,
            wrap_time=None if _is_rapa else CC_LENGTH,
        )

    return _decay_rows


def _cell_decay_constants(
    decays,
    times,
    time_windows,
    wrap_time=None,
):
    _decay_array = np.full(
        (times.shape[0], decays.shape[0]),
        np.nan,
        dtype=float
    )

    if wrap_time is not None:
        times = _wrap_time(times, wrap_time)

    for i, (_left, _right) in enumerate(time_windows):
        _t_idx = np.ones(_decay_array.shape[0], dtype=bool)

        if _left is not None:
            _t_idx &= times >= _left

        if _right is not None:
            _t_idx &= times < _right

        # Put the appropriate decay constants into the array
        _decay_array[_t_idx, :] = decays[:, i].ravel()[None, :]

    return _decay_array
