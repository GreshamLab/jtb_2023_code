import numpy as _np
import pandas as _pd
import matplotlib.colors as _colors
import matplotlib.cm as _cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from seaborn.matrix import dendrogram as _sns_dendrogram

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

from ..figure_constants import (
    POOL_PALETTE,
    EXPT_PALETTE,
    GENE_PALETTE,
    CC_COLS,
    CC_PALETTE,
    GENE_CAT_PALETTE,
    GENE_CAT_COLS
)


def _cmap_to_palette(cmap, n):
    cmap = _cm.get_cmap(cmap, n)
    return [_colors.rgb2hex(cmap(x)) for x in range(n)]


def pool_palette():
    return _cmap_to_palette(POOL_PALETTE, 9)[1:9]


def to_pool_colors(series_data):
    return series_data.map(
        {k: v for k, v in zip(range(1, 9), pool_palette())}
    ).values


def expt_palette(long=False):
    ep = _cmap_to_palette(EXPT_PALETTE, 8)
    return [ep[2], ep[5]] if not long else [ep[2], ep[5], ep[1], ep[6]]


def to_expt_colors(series_data):
    return series_data.map(
        {k: v for k, v in zip(range(1, 3), expt_palette())}
    ).values


def strain_palette():
    ep = _cmap_to_palette(GENE_PALETTE, 8)
    return [ep[0], ep[3]]


def cc_palette():
    return _cmap_to_palette(CC_PALETTE, len(CC_COLS))


def to_cc_colors(series_data):
    return series_data.map(
        {k: v for k, v in zip(CC_COLS, cc_palette())}
    ).values


def gene_category_palette():
    return _cmap_to_palette(GENE_CAT_PALETTE, len(GENE_CAT_COLS))


def squeeze_data(data, high, low=None):
    low = -1 * high if low is None else low
    data = data.copy()
    data[data > high] = high
    data[data < low] = low
    return data


def add_legend_axis(ax, size="8%", pad=0.05, add_extra_pad=None):
    divider = make_axes_locatable(ax)
    lax = divider.append_axes("right", size=size, pad=pad)
    lax.axis("off")

    if add_extra_pad is not None:
        divider.append_axes("right", size=size, pad=pad)

    return lax


def add_legend(ax, colors, labels, title=None, horizontal=False, **kwargs):
    ax.axis("off")
    _ = [ax.scatter([], [], c=c, label=l) for c, l in zip(colors, labels)]
    return ax.legend(
        frameon=False,
        loc="center left",
        ncol=len(labels) if horizontal else 1,
        handletextpad=0.1 if horizontal else 0.8,
        borderpad=0.1,
        borderaxespad=0.1,
        columnspacing=1 if horizontal else 0,
        mode=None,
        title=title,
        **kwargs,
    )


def add_legend_in_plot(
    ax,
    colors,
    labels,
    title=None,
    frameon=True,
    loc="center right",
    bbox_to_anchor=(0, 0.85),
):
    [ax.scatter([], [], c=c, label=l) for c, l in zip(colors, labels)]

    return ax.legend(
        frameon=frameon,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=1,
        columnspacing=0,
        mode=None,
        title=title,
        fontsize=8,
        title_fontsize=8,
        markerscale=0.75,
    )


def plot_heatmap(
    figure,
    matrix_data,
    matrix_cmap,
    matrix_ax,
    dendro_data=None,
    dendro_linkage=None,
    dendro_ax=None,
    row_data=None,
    row_cmap=None,
    row_ax=None,
    row_xlabels=None,
    vmin=None,
    vmax=None,
    add_colorbar=True,
    colorbar_ax=None,
    colorbar_label=None,
    colorbar_loc=None,
):
    refs = {}

    refs["matrix"] = matrix_ax.pcolormesh(
        matrix_data, cmap=matrix_cmap, vmin=vmin, vmax=vmax
    )
    matrix_ax.invert_yaxis()
    matrix_ax.set_xticks([])
    matrix_ax.set_yticks([])
    matrix_ax.spines["right"].set_visible(False)
    matrix_ax.spines["top"].set_visible(False)
    matrix_ax.spines["left"].set_visible(False)
    matrix_ax.spines["bottom"].set_visible(False)

    if row_data is not None:
        _n_row_cols = row_data.shape[1] if row_data.ndim > 1 else 1

        refs["rows"] = row_ax.pcolormesh(row_data, cmap=row_cmap)
        row_ax.invert_yaxis()
        row_ax.set_yticks([])
        row_ax.spines["right"].set_visible(False)
        row_ax.spines["top"].set_visible(False)
        row_ax.spines["left"].set_visible(False)
        row_ax.set_xticks(_np.arange(_n_row_cols) + 0.5)

    if row_xlabels is not None:
        row_ax.set_xticklabels(row_xlabels, rotation=90, va="top", fontsize=6)

    if dendro_ax is not None:
        refs["dendro"] = _sns_dendrogram(
            dendro_data,
            metric="correlation",
            method="average",
            label=False,
            axis=0,
            rotate=True,
            ax=dendro_ax,
            linkage=dendro_linkage,
            tree_kws={"linewidths": 0.05},
        )
        dendro_ax.invert_xaxis()
        dendro_ax.axis("off")

    if add_colorbar:
        vmin = _np.nanmin(matrix_data) if vmin is None else vmin
        vmax = _np.nanmax(matrix_data) if vmax is None else vmax
        refs["cax"] = (
            figure.add_axes(colorbar_loc)
            if colorbar_ax is None
            else colorbar_ax
        )
        refs["cbar"] = figure.colorbar(
            refs["matrix"],
            cax=refs["cax"],
            orientation="vertical",
            ticks=[vmin, vmax]
        )
        refs["cax"].yaxis.set_tick_params(pad=0)
        if vmax >= 100000:
            refs["cax"].set_yticklabels([str(vmin), f"{int(vmax / 1000)}k"])
        elif vmax >= 1000:
            refs["cax"].set_yticklabels([str(vmin), f"{vmax / 1000:.1f}k"])

    if colorbar_label is not None:
        refs["cbar"].set_label(colorbar_label, labelpad=-1)

    return refs


def plot_umap(
    adata,
    ax,
    alpha=1,
    size=1,
    color=None,
    palette=None,
    cmap=None,
    seed=5000,
    **kwargs
):
    rng = _np.random.default_rng(seed)
    overplot = _np.arange(adata.shape[0])
    rng.shuffle(overplot)

    if palette is not None:
        codes, uniques = _pd.factorize(adata.obs[color])
        colors = _np.array(palette)[codes][overplot]
        c = None
    elif color is not None:
        colors = None
        c = adata.obs[color].values[overplot]
    else:
        colors = None
        c = None

    ref = ax.scatter(
        adata.obsm["X_umap"][overplot, 0],
        adata.obsm["X_umap"][overplot, 1],
        color=colors,
        c=c,
        alpha=alpha,
        s=size,
        cmap=cmap,
        **kwargs,
    )

    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_xlabel("UMAP1", size=8)
    ax.set_ylabel("UMAP2", size=8)

    return ref


def velocity_axes(ax):
    ax.spines["left"].set_position(("axes", 0.0))
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["top"].set_color("none")


def ticks_off(ax):
    ax.set_yticks([], [])
    ax.set_xticks([], [])


def shift_axis(ax, x=None, y=None):
    _p = ax.get_position()

    if x is not None:
        _p.x0 += x
        _p.x1 += x
    if y is not None:
        _p.y0 += y
        _p.y1 += y

    ax.set_position(_p)


def cluster_on_rows(dataframe, **kwargs):
    return dataframe.index[
        dendrogram(
            linkage(
                pdist(dataframe.values, **kwargs),
                method="ward"
            ),
            no_plot=True
        )["leaves"]
    ]


def plot_stacked_barplot(dataframe, ax, stack_order, palette=None):
    ref = {}

    _bottoms = _np.zeros(dataframe.shape[0])
    _x = _np.arange(dataframe.shape[0])

    for c, cc in enumerate(stack_order):
        _cat_data = dataframe[cc].values
        _cat_color = palette[c] if palette is not None else None

        ref[cc] = ax.bar(
            _x,
            _cat_data,
            label=cc,
            bottom=_bottoms,
            color=_cat_color
        )

        _bottoms += _cat_data

    return ref


def symmetric_ylim(ax, integer=False, one_decimal=True, lim_max=None):
    _comp_ylim = max(map(abs, ax.get_ylim()))
    _comp_ylim = max(_comp_ylim, 0.1)

    if lim_max is not None:
        _comp_ylim = min(_comp_ylim, lim_max)

    if one_decimal:
        _tick = int(_comp_ylim * 10) / 10
    elif integer:
        _tick = int(_comp_ylim)

    ax.set_ylim(-1 * _comp_ylim, _comp_ylim)
    ax.set_yticks([-1 * _tick, 0, _tick], [-1 * _tick, 0, _tick], size=8)


def align_ylim(ax, ax2):
    ax2.set_ylim(*ax.get_ylim())
    ax2.set_yticks(ax.get_yticks(), ax.get_yticks(), size=8)


def plot_correlations(
    x,
    ax,
    plot_index=None,
    cmap="bwr",
    vmin=-1,
    vmax=1,
    cat_ax=None,
    cat_cmap=None,
    cat_var=None,
):
    _corr = _np.corrcoef(x.T)
    _corr_finite = ~_np.all(_np.isnan(_corr), axis=0)
    _corr = _corr[_corr_finite, :][:, _corr_finite]

    if plot_index is None:
        _corr_idx = dendrogram(
            linkage(
                squareform(1 - _corr, checks=False),
                method="average"
            ),
            no_plot=True
        )["leaves"]
    else:
        _corr_idx = plot_index

    ref = ax.pcolormesh(
        _np.tril(_corr[_corr_idx, :][:, _corr_idx])[::-1, :],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks([], [])
    ax.set_yticks([], [])

    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    if cat_ax is not None:
        cat_ax.pcolormesh(
            cat_var[_corr_finite][_corr_idx].reshape(-1, 1)[::-1, :],
            cmap=cat_cmap
        )
        _cat_col_axis(cat_ax)

    return ref, _corr_idx


def _cat_col_axis(ax):
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0.5], [])
