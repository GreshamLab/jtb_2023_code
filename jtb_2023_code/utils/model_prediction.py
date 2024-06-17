import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import gc

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from inferelator_velocity.utils.aggregation import (
    aggregate_sliding_window_times
)

from jtb_2023_code.utils.figure_common import (
    pool_palette,
    to_pool_colors,
    velocity_axes
)
from jtb_2023_code.utils.figure_data import common_name, FigureSingleCellData
from jtb_2023_code.figure_constants import (
    SUPIRFACTOR_COUNT_MODEL,
    SUPIRFACTOR_VELOCITY_DYNAMICAL_MODEL,
    SUPIRFACTOR_BIOPHYSICAL_MODEL
)

from supirfactor_dynamical import (
    read,
    TimeDataset,
    TruncRobustScaler,
    predict_perturbation
)


def predict_all(
    data,
    data_processed=False,
    untreated_only=True,
    n_predicts=60,
    predict_times=None
):
    count_model = (
        read(SUPIRFACTOR_COUNT_MODEL)
        .eval()
        .set_time_parameters(n_additional_predictions=0, loss_offset=0)
    )
    velo_model = (
        read(SUPIRFACTOR_VELOCITY_DYNAMICAL_MODEL)
        .eval()
        .set_time_parameters(n_additional_predictions=0, loss_offset=0)
    )
    biophysics_model = (
        read(SUPIRFACTOR_BIOPHYSICAL_MODEL)
        .eval()
        .set_time_parameters(n_additional_predictions=0, loss_offset=0)
    )

    _count_scaler = biophysics_model._count_inverse_scaler.numpy()
    _velo_scaler = biophysics_model._velocity_inverse_scaler.numpy()

    if data_processed:
        model_data = data
        model_scaler = None
    else:
        model_data, model_scaler = process_data_for_model(
            data,
            count_model.prior_network_labels[0],
            scale_factor=_count_scaler
        )

    g = model_data.shape[1]

    print("Predicting From Count Model")

    predicts = _initialize_adata(
        np.multiply(
            predict_from_model(
                count_model,
                model_data,
                untreated_only=untreated_only,
                n_predicts=n_predicts,
            ),
            _count_scaler[None, None, :],
        ).reshape(-1, g),
        model_data,
        untreated_only=untreated_only,
        n_predicts=n_predicts,
        predict_times=predict_times,
    )

    predicts.var['count_scale'] = _count_scaler
    predicts.var['velocity_scale'] = _velo_scaler

    predicts.layers["count_predict_counts"] = predicts.X.copy()

    print("Predicting From Velocity Model")

    _velo_predicts = predict_from_model(
        velo_model,
        model_data,
        untreated_only=untreated_only,
        n_predicts=n_predicts,
    )
    predicts.layers["velocity_predict_counts"] = np.multiply(
        _velo_predicts[1],
        _count_scaler[None, None, :],
    ).reshape(-1, g)

    predicts.layers["velocity_predict_velocity"] = np.multiply(
        _velo_predicts[0],
        _velo_scaler[None, None, :],
    ).reshape(-1, g)

    del _velo_predicts

    predict_biophysics(
        model_data,
        predicts,
        untreated_only=untreated_only,
        n_predicts=n_predicts,
        predict_times=predict_times,
    )

    return model_data, predicts, model_scaler


def predict_biophysics(
    model_data,
    predicts=None,
    untreated_only=True,
    n_predicts=60,
    predict_times=None
):
    print("Predicting From Biophysical Model")
    biophysics_model = read(SUPIRFACTOR_BIOPHYSICAL_MODEL).eval()
    _count_scaler = biophysics_model._count_inverse_scaler.numpy()
    _velo_scaler = biophysics_model._velocity_inverse_scaler.numpy()
    g = model_data.shape[1]
   
    _velo_predict = np.multiply(
        predict_from_model(
            biophysics_model,
            model_data,
            untreated_only=untreated_only,
            n_predicts=n_predicts,
        )[0],
        _velo_scaler[None, None, :],
    ).reshape(-1, g)

    if predicts is None:
        predicts = _initialize_adata(
            _velo_predict,
            model_data,
            untreated_only=untreated_only,
            n_predicts=n_predicts,
            predict_times=predict_times,
        )
        predicts.layers["biophysical_predict_velocity"] = predicts.X.copy()

    else:
        predicts.layers["biophysical_predict_velocity"] = _velo_predict

    del _velo_predict

    _sub_predicts, counts, decays, tfa = predict_from_model(
        biophysics_model,
        model_data,
        return_submodels=True,
        untreated_only=untreated_only,
        n_predicts=n_predicts,
    )

    predicts.layers["biophysical_predict_transcription"] = np.multiply(
        _sub_predicts[0], _velo_scaler[None, None, :]
    ).reshape(-1, g)

    predicts.layers["biophysical_predict_decay"] = np.multiply(
        _sub_predicts[1], _velo_scaler[None, None, :]
    ).reshape(-1, g)

    predicts.layers["biophysical_predict_counts"] = np.multiply(
        counts, _count_scaler[None, None, :]
    ).reshape(-1, g)

    predicts.layers["biophysical_predict_decay_constant"] = np.divide(
        decays, _count_scaler[None, None, :]
    ).reshape(-1, g)

    predicts.obsm["biophysical_predict_tfa"] = tfa.reshape(-1, tfa.shape[-1])

    return predicts


def _initialize_adata(
    predict_data,
    model_data,
    untreated_only=True,
    n_predicts=60,
    predict_times=None
):
    predicts = ad.AnnData(predict_data)
    predicts.var_names = model_data.var_names.copy()

    _add_predict_times(
        predicts,
        untreated_only=untreated_only,
        n_predicts=n_predicts,
        predict_times=predict_times,
    )

    return predicts


def _add_predict_times(
    predicts, untreated_only=True, n_predicts=60, predict_times=None
):
    if untreated_only:
        _n = n_predicts
    else:
        _n = n_predicts + 60

    if predict_times is None:
        predict_times = np.arange(-10, _n) + 0.5

    predicts.obs["program_rapa_time"] = np.tile(
        predict_times, int(predicts.shape[0] / predict_times.shape[0])
    )

    predicts.obs["color"] = pool_palette()[1]

    for i in range(6):
        _idx = predicts.obs["program_rapa_time"] > i * 10
        _idx &= predicts.obs["program_rapa_time"] < (i + 1) * 10
        predicts.obs.loc[_idx, "color"] = pool_palette()[i + 2]

    return predicts


def process_data_for_model(data, genes=None, wt_only=True, scale=True, scale_factor=None):
    if wt_only:
        _idx = data.obs["Gene"] == "WT"
    else:
        _idx = np.ones(data.shape[0], dtype=bool)

    model_data = ad.AnnData(data.layers["counts"][_idx, :])
    model_data.obs = data.obs.loc[_idx, :].copy()

    if "program_1_time" in model_data.obs.columns:
        model_data.obs["program_rapa_time"] = model_data.obs["program_1_time"]

    model_data.var = data.var.copy()

    model_data = FigureSingleCellData._normalize(
        model_data,
        method='depth',
        n_counts=2000
    )
    
    if genes is not None and (
        (len(genes) != model_data.shape[1]) or not
        all(model_data.var_names == genes)
    ):
        model_data = model_data[:, genes].copy()

    return _process_for_model(model_data, scale=scale, scale_factor=scale_factor)


def process_velocity_for_model(data_obj, genes=None, scale=True, scale_factor=None):
    _velo = []
    _obs = []

    for i in range(1, 3):
        dd = data_obj.decay_data(i, "WT")
        _velo.append(
            dd.layers["rapamycin_velocity"] + dd.layers["cell_cycle_velocity"]
        )
        _obs.append(dd.obs.copy())

        del dd
        gc.collect()

    model_data = ad.AnnData(
        np.vstack(_velo),
        var=data_obj.all_data.var,
        obs=pd.concat(_obs).reset_index(drop=True)
    )

    del _velo
    gc.collect()

    if genes is not None:
        model_data = model_data[:, genes].copy()

    return _process_for_model(model_data, scale=scale, scale_factor=scale_factor)


def _process_for_model(model_data, scale=True, scale_factor=None):
    model_data.obs["Test"] = _get_test_idx(model_data.shape[0])

    if scale:
        data_scaler = TruncRobustScaler(with_centering=False)

        if scale_factor is None:
            model_data.layers["scaled"] = data_scaler.fit_transform(model_data.X)
        else:
            data_scaler.scale_ = scale_factor
            model_data.layers["scaled"] = data_scaler.transform(model_data.X)
            
        model_data.var["scale"] = data_scaler.scale_

    else:
        data_scaler = None

    return model_data, data_scaler


def _get_test_idx(n):
    _, test_idx = train_test_split(
        np.arange(n),
        test_size=0.25,
        random_state=1800
    )

    _test = np.zeros(n, dtype=bool)
    _test[test_idx] = True

    return _test


def predict_perturbation_from_model(
    fit_model,
    model_data,
    perturbation,
    layer="scaled",
    n_predicts=60,
    **kwargs
):
    data = _to_dataloader(model_data, layer=layer, untreated_only=True)

    return _to_numpy(
        predict_perturbation(
            fit_model,
            data,
            perturbation,
            n_predicts,
            unmodified_counts=False,
            **kwargs
        )
    )


def predict_from_model(
    fit_model,
    model_data,
    untreated_only=True,
    return_data_stacked=False,
    layer="scaled",
    n_predicts=60,
    **kwargs
):
    data = _to_dataloader(
        model_data,
        layer=layer,
        untreated_only=untreated_only,
        stack_data=return_data_stacked,
    )

    if untreated_only:
        kwargs["n_time_steps"] = n_predicts

    predicts_training = _to_numpy(
        fit_model(
            data[..., 0] if return_data_stacked else data,
            **kwargs
        ),
        add_dim=return_data_stacked,
    )

    if return_data_stacked:
        data = data.numpy()

        if isinstance(predicts_training, tuple):
            predicts_training = (data,) + predicts_training
        else:
            predicts_training = (data, predicts_training)

        predicts_training = np.concatenate(predicts_training, 3)

    return predicts_training


def _to_dataloader(
    model_data,
    layer="scaled",
    untreated_only=True,
    stack_data=False
):
    test_idx = model_data.obs["Test"].values

    if layer == "X":
        _data = model_data.X[test_idx, :]

    else:
        _data = model_data.layers[layer][test_idx, :]

    try:
        _data = _data.A
    except AttributeError:
        pass

    if stack_data and "velocity" in model_data.layers:
        _data = np.stack(
            (
                _data,
                model_data.layers["velocity"][test_idx, :]
            ),
            -1
        )
    elif stack_data:
        _data = np.stack(
            (
                _data,
                model_data.layers["rapamycin_velocity"][test_idx, :] +
                model_data.layers["cell_cycle_velocity"][test_idx, :],
            ),
            -1,
        )

    return next(
        iter(
            DataLoader(
                TimeDataset(
                    _data,
                    model_data.obs["program_rapa_time"].values[test_idx],
                    -10,
                    0 if untreated_only else 60,
                    1,
                    sequence_length=10 if untreated_only else 70,
                    shuffle_time_vector=[-10, 0],
                ),
                batch_size=1000,
                drop_last=False,
            )
        )
    )


def _to_numpy(x, add_dim=False):
    if isinstance(x, (tuple, list)):
        return tuple(_to_numpy(y) for y in x)

    if x is None:
        return None

    if x.requires_grad:
        x = x.detach()

    if add_dim:
        return x.numpy()[..., np.newaxis]

    else:
        return x.numpy()


def plot_gene(
    d,
    gene,
    ax,
    rgen,
    layer="X",
    velocity=False,
    predicts=False,
    test_only=False,
    time_positive_only=False,
    alpha=0.02,
    annotation_loc=(0.65, 0.6),
    gene_data_hook=None,
    predicts_color='red'
):
    fig_refs = {}

    if test_only:
        test_idx = d.obs["Test"].values
    else:
        test_idx = np.ones(d.shape[0], dtype=bool)

    if time_positive_only:
        test_idx &= d.obs["program_rapa_time"] > 0

    _time = d.obs["program_rapa_time"].values[test_idx]

    if predicts:
        _color = np.array([predicts_color] * d.shape[0])
    else:
        _color = to_pool_colors(d.obs["Pool"]).astype(str)

    _color = _color[test_idx]

    if layer == "X":
        _gene_data = d.X[:, d.var_names.get_loc(gene)][test_idx]
    else:
        _gene_data = d.layers[layer][:, d.var_names.get_loc(gene)][test_idx]

    overplot_shuffle = np.arange(len(_time))
    rgen.shuffle(overplot_shuffle)

    try:
        _gene_data = _gene_data.A
    except AttributeError:
        pass

    _gene_data = _gene_data.ravel()

    if time_positive_only:
        _centers = np.linspace(0 + 0.5, 60 - 0.5, 60)
    else:
        _centers = np.linspace(-10 + 0.5, 60 - 0.5, 70)

    median_counts, _window_centers = aggregate_sliding_window_times(
        _gene_data.reshape(-1, 1),
        _time,
        centers=_centers,
        width=1.0,
        agg_func=np.median,
    )

    if gene_data_hook is not None:
        _gene_data = gene_data_hook(_gene_data)
        median_counts = gene_data_hook(median_counts)

    fig_refs["scatter"] = ax.scatter(
        x=_time[overplot_shuffle],
        y=_gene_data[overplot_shuffle],
        c=_color[overplot_shuffle],
        alpha=alpha,
        s=1,
    )

    if annotation_loc is not None:
        fig_refs["annotate"] = ax.annotate(
            common_name(gene),
            annotation_loc,
            xycoords="axes fraction",
            color="black",
            size=7,
            fontstyle="italic",
            bbox=dict(
                facecolor="white",
                alpha=0.5,
                edgecolor="white",
                boxstyle="round"
            ),
        )

    fig_refs["median_plot"] = ax.plot(
        _window_centers,
        median_counts,
        c="black",
        alpha=0.75,
        linewidth=1.0,
        linestyle="solid" if not predicts else "dashed",
        zorder=2,
    )

    ax.set_xlim(-10, 65)
    ax.set_xticks([0, 30, 60], [])

    if predicts:
        pass
    elif velocity:
        _ylim = np.quantile(np.abs(_gene_data), 0.995)
        ax.set_ylim(-1 * _ylim, _ylim)
        velocity_axes(ax)
    else:
        ax.set_ylim(0, np.quantile(_gene_data, 0.995))

    if not predicts:
        ax.axvline(0, 0, 1, linestyle="--", linewidth=1.0, c="black")

    ax.tick_params(axis="both", which="major", labelsize=8)

    return fig_refs
