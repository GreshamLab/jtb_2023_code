import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def plot_pca(
    ax,
    data,
    obsm_key='X_pca',
    uns_key='pca',
    pcs=(0, 1),
    row_mask=None,
    label_size=10,
    **kwargs
):
    ref = _plot(
        ax,
        data,
        obsm_key=obsm_key,
        cols=pcs,
        row_mask=row_mask,
        **kwargs
    )

    if uns_key is not None:
        _vr = data.uns[uns_key]['variance_ratio']
    else:
        _vr = [0] * (max(pcs) + 1)

    ax.set_xlabel(
        f"PC{pcs[0] + 1} ({_vr[pcs[0]] * 100:.1f}%)",
        size=label_size
    )
    ax.set_ylabel(
        f"PC{pcs[1] + 1} ({_vr[pcs[1]] * 100:.1f}%)",
        size=label_size
    )
    return ref


def plot_umap(
    ax,
    data,
    obsm_key='X_umap',
    row_mask=None,
    label_size=10,
    **kwargs
):
    ref = _plot(
        ax,
        data,
        obsm_key=obsm_key,
        row_mask=row_mask,
        **kwargs
    )
    ax.set_xlabel("UMAP1", size=label_size)
    ax.set_ylabel("UMAP2", size=label_size)
    return ref


def _plot(
    ax,
    data,
    cols=(0, 1),
    use_varm=False,
    obsm_key="X_pca",
    row_mask=None,
    seed=43,
    c=None,
    alpha=None,
    **kwargs
):

    if use_varm:
        n = data.shape[1]
        dref = data.varm

    else:
        n = data.shape[0]
        dref = data.obsm

    rng = np.random.default_rng(seed)
    overplot = np.arange(n)

    if row_mask is not None:
        overplot = overplot[row_mask]

    rng.shuffle(overplot)

    if isinstance(c, (np.ndarray, pd.Index)):
        c = c[overplot]
    elif isinstance(c, pd.Series):
        c = c.iloc[overplot]

    if isinstance(alpha, (np.ndarray, pd.Index)):
        alpha = alpha[overplot]
    elif isinstance(alpha, pd.Series):
        alpha = alpha.iloc[overplot]

    ref = ax.scatter(
        dref[obsm_key][overplot, cols[0]],
        dref[obsm_key][overplot, cols[1]],
        c=c,
        alpha=alpha,
        **kwargs
    )
    ax.set_xticks([], [])
    ax.set_yticks([], [])

    return ref


def add_legend(
    ax,
    colors,
    labels,
    loc='upper right',
    fontsize=8,
    alpha=1,
    scatter_kwargs={},
    **kwargs
):
    [
        ax.scatter([], [], color=c, label=l, alpha=alpha, **scatter_kwargs)
        for c, l in zip(colors, labels)
    ]
    return ax.legend(loc=loc, fontsize=fontsize, **kwargs)


def add_colorbar(
    ax,
    cmap,
    min=0,
    max=1,
    fontsize=6,
    orientation='vertical',
    **kwargs
):

    _ref = plt.colorbar(
        matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(0, 1),
            cmap=cmap
        ),
        cax=ax,
        orientation=orientation,
        **kwargs
    )

    if orientation == 'vertical':
        ax.set_yticks([0, 1], [min, max], size=fontsize)
    else:
        ax.set_xticks([0, 1], [min, max], size=fontsize)

    return _ref


def umap_corner_label(ax, ratio=0.3):
    _xmin, _xmax = ax.get_xlim()
    _ymin, _ymax = ax.get_ylim()

    _xaxis_right = (_xmax - _xmin) * ratio + _xmin
    _yaxis_top = (_ymax - _ymin) * ratio + _ymin

    ax.spines['bottom'].set_bounds(ax.get_xlim()[0], _xaxis_right)
    ax.spines['left'].set_bounds(ax.get_ylim()[0], _yaxis_top)
    ax.spines[['top', 'right']].set_visible(False)

    ax.set_xlabel('UMAP1', size=6, loc='left')
    ax.set_ylabel('UMAP2', size=6, loc='bottom')


def lims_for_rescale_umap_keep_aspect(
    adata,
    idx,
    obsm='X_umap',
    pad=0.5,
    quantiles=(0.005, 0.995)
):
    _obsm = adata.obsm[obsm]

    x_lim = np.nanquantile(
        _obsm[idx, 0], quantiles
    )
    y_lim = np.nanquantile(
        _obsm[idx, 1], quantiles
    )
    _old_xlim = (np.nanmin(_obsm[:, 0]), np.nanmax(_obsm[:, 0]))
    _old_ylim = (np.nanmin(_obsm[:, 1]), np.nanmax(_obsm[:, 1]))
    
    new_range = np.array([x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]])
    old_range = np.array([_old_xlim[1] - _old_xlim[0], _old_ylim[1] - _old_ylim[0]])
    
    new_range = old_range * np.max(new_range / old_range) / 2 + pad
    new_x = (np.mean(x_lim) - new_range[0], (np.mean(x_lim) + new_range[0]))
    new_y = (np.mean(y_lim) - new_range[1], (np.mean(y_lim) + new_range[1]))

    return new_x, new_y
