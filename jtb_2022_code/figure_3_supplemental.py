import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

from jtb_2022_code.utils.figure_common import *
from jtb_2022_code.figure_constants import *
from .figure_3 import _get_fig3_data, _fig3_plot

from inferelator_velocity.decay import calc_decay
from matplotlib import collections
from sklearn.linear_model import LinearRegression


def figure_3_supplement_1_plot(data, save=True):

    fig_refs = {}

    fig = plt.figure(figsize=(6, 4), dpi=300)

    _width = 0.3
    _halfwidth = 0.12
    _height = 0.22

    _x_left = 0.12
    _x_right = 0.55

    axd = {
        'exp_var_1': fig.add_axes([_x_left + 0.05, 0.72, 0.2, 0.2]),
        'exp_var_2': fig.add_axes([0.6, 0.72, 0.2, 0.2]),

        'pc12_1': fig.add_axes([_x_left, 0.36, _halfwidth, _height]),
        'denoised_pc12_1': fig.add_axes([_x_left + _halfwidth + 0.05, 0.36, _halfwidth, _height]),

        'pc12_2': fig.add_axes([_x_right, 0.36, _halfwidth, _height]),
        'denoised_pc12_2': fig.add_axes([_x_right + _halfwidth + 0.05, 0.36, _halfwidth, _height]),

        'pc13_1': fig.add_axes([_x_left, 0.08, _halfwidth, _height]),
        'denoised_pc13_1': fig.add_axes([_x_left + _halfwidth + 0.05, 0.08, _halfwidth, _height]),

        'pc13_2': fig.add_axes([_x_right, 0.08, _halfwidth, _height]),
        'denoised_pc13_2': fig.add_axes([_x_right + _halfwidth + 0.05, 0.08, _halfwidth, _height]),

        'legend': fig.add_axes([0.9, 0.1, 0.1, 0.55])
    }

    for i in range(1, 3):

        expt_dd = data.denoised_data(i, "WT")
        expt_dd.obsm['X_pca'] = expt_dd.obsm['denoised_pca']
        _n_pcs = 100

        rgen = np.random.default_rng(441)
        overplot_shuffle = np.arange(expt_dd.shape[0])
        rgen.shuffle(overplot_shuffle)

        fig_refs[f'exp_var_{i}'] = axd[f'exp_var_{i}'].plot(
            np.arange(0, _n_pcs + 1), 
            np.insert(np.cumsum(data.expt_data[(i, "WT")].uns['pca']['variance_ratio'][:_n_pcs]), 0, 0),
            c='black',
            alpha=1
        )

        fig_refs[f'exp_var_{i}'] = axd[f'exp_var_{i}'].plot(
            np.arange(0, _n_pcs + 1), 
            np.insert(np.cumsum(expt_dd.uns['pca']['variance_ratio'][:_n_pcs]), 0, 0),
            c='black',
            linestyle='--',
            alpha=1
        )

        axd[f'exp_var_{i}'].set_title(f"Replicate {i}", size=8)
        axd[f'exp_var_{i}'].set_ylim(0, 1)
        axd[f'exp_var_{i}'].set_xlim(0, 100)
        axd[f'exp_var_{i}'].set_xlabel("# PCs", size=8)
        axd[f'exp_var_{i}'].set_ylabel("Cum. Var. Expl.", size=8)
        axd[f'exp_var_{i}'].tick_params(labelsize=8)

        for k in range(2, 4):
            comp_str = str(1) + ',' + str(k)
            sc.pl.pca(
                data.expt_data[(i, "WT")],
                ax=axd[f"pc1{k}_{i}"],
                components=comp_str,
                color='Pool',
                palette= pool_palette(),
                title=None,
                show=False,
                alpha=0.25,
                size=2,
                legend_loc='none',
                annotate_var_explained=True
            )

            axd[f"pc1{k}_{i}"].set_title(None)
            axd[f"pc1{k}_{i}"].xaxis.label.set_size(8)
            axd[f"pc1{k}_{i}"].yaxis.label.set_size(8)

            sc.pl.pca(
                expt_dd,
                ax=axd[f"denoised_pc1{k}_{i}"],
                components=comp_str,
                color='Pool',
                palette= pool_palette(),
                title=None,
                show=False,
                alpha=0.25,
                size=2,
                legend_loc='none',
                annotate_var_explained=True
            )

            axd[f"denoised_pc1{k}_{i}"].set_title(None)
            axd[f"denoised_pc1{k}_{i}"].xaxis.label.set_size(8)
            axd[f"denoised_pc1{k}_{i}"].yaxis.label.set_size(8)

            if k == 2:
                axd[f"pc1{k}_{i}"].set_title("Counts", size=8)
                axd[f"denoised_pc1{k}_{i}"].set_title("Denoised", size=8)

    axd['legend'].imshow(plt.imread(FIG_RAPA_LEGEND_VERTICAL_FILE_NAME), aspect='equal')
    axd['legend'].axis('off')
    
    axd[f'exp_var_1'].set_title("A", loc='left', x=-0.6, y=0.9, weight='bold')
    axd[f'pc12_1'].set_title("B", loc='left', x=-0.4, y=0.9, weight='bold')
    
    if save:
        fig.savefig(FIGURE_3_SUPPLEMENTAL_FILE_NAME + "_1.png", facecolor='white')

    return fig

    
def figure_3_supplement_2_plot(data, save=True):
    
    # GET DATA #
    expt2 = data.expt_data[(2, "WT")]
    expt2_denoised_X = data.denoised_data(2, "WT").X
    
    # MAKE FIGURE #
    fig = plt.figure(figsize=(5, 8), dpi=SUPPLEMENTAL_FIGURE_DPI)

    axd = {
        'image': fig.add_axes([0.075, 0.725, 0.85, 0.225]),
        'pca_1': fig.add_axes([0.075, 0.45, 0.35, 0.25]),
        'velo_1': fig.add_axes([0.60, 0.45, 0.35, 0.25]),
        'pca_2': fig.add_axes([0.075, 0.1, 0.35, 0.25]),
        'velo_2': fig.add_axes([0.60, 0.1, 0.35, 0.25])
    }

    # PLOT ONE EXAMPLE #
    _plot_velocity_calc(
        expt2,
        expt2_denoised_X,
        expt2.obs.index.get_loc(expt2.obs.sample(1, random_state=100).index[0]),
        pca_ax=axd['pca_1'],
        expr_ax=axd['velo_1'],
        gene="YOR063W",
        gene_name=data.gene_common_name("YOR063W"),
        time_obs_key=f"program_{data.all_data.uns['programs']['rapa_program']}_time"
    )

    # PLOT ANOTHER EXAMPLE #
    _plot_velocity_calc(
        expt2,
        expt2_denoised_X,
        expt2.obs.index.get_loc(expt2.obs.sample(1, random_state=101).index[0]),
        pca_ax=axd['pca_2'],
        expr_ax=axd['velo_2'],
        gene="YKR039W",
        gene_name=data.gene_common_name("YKR039W"),
        time_obs_key=f"program_{data.all_data.uns['programs']['rapa_program']}_time"
    )

    axd['pca_1'].set_title(r"$\bf{B}$" + "  i", loc='left', x=-0.2)
    axd['pca_2'].set_title(r"$\bf{C}$" + "  i", loc='left', x=-0.2)
    axd['velo_1'].set_title("ii", loc='left', x=-0.2)
    axd['velo_2'].set_title("ii", loc='left', x=-0.2)

    # DRAW SCHEMATIC #
    axd['image'].set_title(r"$\bf{A}$", loc='left', x=-0.075)
    axd['image'].imshow(plt.imread(SFIG3A_FILE_NAME), aspect='equal')
    axd['image'].axis('off')

    if save:
        fig.savefig(FIGURE_3_SUPPLEMENTAL_FILE_NAME + "_2.png", facecolor='white')
    
    return fig

def figure_3_supplement_3_plot(data, f3_data=None, save=True):
    
    if f3_data is None:
        f3_data = _get_fig3_data(data)
        
    time_key = f"program_{data.all_data.uns['programs']['rapa_program']}_time"
    
    def ticks_off(ax):
        ax.set_yticks([], [])
        ax.set_xticks([], [])

    fig_refs = {}

    fig = plt.figure(figsize=(4, 4), dpi=300)

    _small_w = 0.1
    _small_h = 0.125

    axd = {
        'image': fig.add_axes([0.15, 0.67, 0.77, 0.33]),
        'decay_1': fig.add_axes([0.2, 0.39, 0.32, 0.23]),
        'decay_2': fig.add_axes([0.60, 0.39, 0.32, 0.23]),
        'decay_1-1': fig.add_axes([0.2, 0.16, _small_w, _small_h]),
        'decay_1-2': fig.add_axes([0.31, 0.16, _small_w, _small_h]),
        'decay_1-3': fig.add_axes([0.42, 0.16, _small_w, _small_h]),
        'decay_1-4': fig.add_axes([0.2, 0.025, _small_w, _small_h]),
        'decay_1-5': fig.add_axes([0.31, 0.025, _small_w, _small_h]),
        'decay_1-6': fig.add_axes([0.42, 0.025, _small_w, _small_h]),
        'decay_2-1': fig.add_axes([0.6, 0.16, _small_w, _small_h]),
        'decay_2-2': fig.add_axes([0.71, 0.16, _small_w, _small_h]),
        'decay_2-3': fig.add_axes([0.82, 0.16, _small_w, _small_h]),
        'decay_2-4': fig.add_axes([0.6, 0.025, _small_w, _small_h]),
        'decay_2-5': fig.add_axes([0.71, 0.025, _small_w, _small_h]),
        'decay_2-6': fig.add_axes([0.82, 0.025, _small_w, _small_h]),
    }

    color_vec = to_pool_colors(f3_data.obs['Pool'])

    rgen = np.random.default_rng(8)
    overplot_shuffle = np.arange(f3_data.shape[0])
    rgen.shuffle(overplot_shuffle)

    for i, g in enumerate(FIGURE_4_GENES):

        fig_refs[f'decay_{i}'] = axd[f'decay_{i+1}'].scatter(
            x=f3_data.layers['denoised'][overplot_shuffle, i], 
            y=f3_data.layers['velocity'][overplot_shuffle, i],
            c=color_vec[overplot_shuffle],
            alpha=0.2,
            s=1
        )

        xlim = np.quantile(f3_data.layers['denoised'][:, i], 0.995)
        ylim = np.abs(np.quantile(f3_data.layers['velocity'][:, i], [0.001, 0.999])).max()

        axd[f'decay_{i+1}'].set_xlim(0, xlim)
        axd[f'decay_{i+1}'].set_ylim(-1 * ylim, ylim)
        velocity_axes(axd[f'decay_{i+1}'])
        axd[f'decay_{i+1}'].tick_params(labelsize=8)

        if i == 0:
            axd[f'decay_{i+1}'].set_ylabel(
                "RNA Velocity\n(Counts/minute)", size=8
            )

        axd[f'decay_{i+1}'].set_xlabel(
            "Expression (Counts)", size=8, labelpad=20
        )
        axd[f'decay_{i+1}'].set_title(
            data.gene_common_name(g),
            size=8,
            fontdict={'fontweight': 'bold', 'fontstyle': 'italic'}
        )

        dc = calc_decay(
            f3_data.layers['denoised'][:, i].reshape(-1, 1),
            f3_data.layers['velocity'][:, i].reshape(-1, 1),
            include_alpha=False,
            lstatus=False
        )[0][0]

        axd[f'decay_{i+1}'].axline(
            (0, 0),
            slope=-1 * dc,
            color='darkred',
            linewidth=1,
            linestyle='--'
        )

        axd[f'decay_{i+1}'].annotate(
            r"$\lambda$" + f' = {dc:0.3f}',
            xy=(0.05, 0.05),
            xycoords='axes fraction',
            size=6
        )

        for j in range (1, 7):

            if j == 1:
                _start = -10
            else:
                _start = 10 * (j - 1)

            _window_idx = f3_data.obs[time_key] > _start
            _window_idx &= f3_data.obs[time_key] <= (_start + 10)

            _window_adata = f3_data[_window_idx, :]
            _w_overplot = np.arange(_window_adata.shape[0])
            rgen.shuffle(_w_overplot)

            fig_refs[f'decay_{i}_{j}'] = axd[f'decay_{i+1}-{j}'].scatter(
                x=_window_adata.layers['denoised'][_w_overplot, i], 
                y=_window_adata.layers['velocity'][_w_overplot, i],
                c=color_vec[_window_idx][_w_overplot],
                alpha=0.2,
                s=0.5
            )

            dc = calc_decay(
                _window_adata.layers['denoised'][:, i].reshape(-1, 1),
                _window_adata.layers['velocity'][:, i].reshape(-1, 1),
                include_alpha=False,
                lstatus=False
            )[0][0]

            axd[f'decay_{i+1}-{j}'].axline(
                (0, 0),
                slope=-1 * dc,
                color='darkred',
                linewidth=1,
                linestyle='--'
            )

            axd[f'decay_{i+1}-{j}'].annotate(
                r"$\lambda$" + f' = {dc:0.3f}',
                xy=(0.05, 0.05),
                xycoords='axes fraction',
                size=6
            )

            velocity_axes(axd[f'decay_{i+1}-{j}'])
            ticks_off(axd[f'decay_{i+1}-{j}'])

            axd[f'decay_{i+1}-{j}'].set_xlim(0, xlim)
            axd[f'decay_{i+1}-{j}'].set_ylim(-1 * ylim, ylim)

    axd['image'].imshow(plt.imread(SFIG3B_FILE_NAME), aspect='equal')
    axd['image'].axis('off')

    axd['image'].set_title("A", loc='left', weight='bold', x=-0.28, y=0.8)
    axd['decay_1'].set_title("B", loc='left', weight='bold', x=-0.4, y=0.9)
    axd['decay_1-1'].set_title("C", loc='left', weight='bold', x=-1.2, y=0.5)

    if save:
        fig.savefig(FIGURE_3_SUPPLEMENTAL_FILE_NAME + "_3.png", facecolor='white')
        
    return fig


def figure_3_supplement_4_plot(data, f3_data=None, save=True):
    
    if f3_data is None:
        f3_data = _get_fig3_data(data)

    color_vec = to_expt_colors(f3_data.obs['Experiment'])
    
    fig = _fig3_plot(
        f3_data,
        color_vec,
        FIGURE_4_GENES,
        gene_labels=[data.gene_common_name(g) for g in FIGURE_4_GENES],
        time_key=f"program_{data.all_data.uns['programs']['rapa_program']}_time"
    )
    
    if save:
        fig.savefig(FIGURE_3_SUPPLEMENTAL_FILE_NAME + "_4.png", facecolor='white')

    return fig


def _plot_velocity_calc(
        adata,
        expr_layer,
        center,
        pca_ax=None,
        expr_ax=None,
        time_obs_key='program_0_time',
        gene="YKR039W",
        gene_name=None
    ):

    if pca_ax is None or expr_ax is None:
        fig, axs = plt.subplots(1, 2, figsize=(5, 3), dpi=300)
        pca_ax = axs[0]
        expr_ax = axs[1]

    else:
        fig = None

    connecteds = adata.obsp['noise2self_distance_graph'][center, :].nonzero()[1]

    pca_ax.scatter(
        adata.obsm['X_pca'][:, 0],
        adata.obsm['X_pca'][:, 1],
        c=to_pool_colors(adata.obs['Pool']),
        s=0.25,
        alpha=0.1,
        zorder=-1
    )

    pca_ax.add_collection(
        collections.LineCollection(
            [(adata.obsm['X_pca'][center, 0:2], adata.obsm['X_pca'][n, 0:2]) for n in connecteds],
            colors='black',
            linewidths=0.5,
            alpha=0.25
        )
    )

    pca_ax.scatter(
        adata.obsm['X_pca'][connecteds, 0],
        adata.obsm['X_pca'][connecteds, 1],
        s=0.25,
        alpha=0.5,
        c='black'
    )

    pca_ax.scatter(
        adata.obsm['X_pca'][center, 0],
        adata.obsm['X_pca'][center, 1],
        facecolors='none',
        edgecolors='r',
        linewidths=0.5,
        s=4,
        zorder=5
    )

    pca_ax.set_xlabel(f"PC1 ({adata.uns['pca']['variance_ratio'][0] * 100:.2f}%)", size=8)
    pca_ax.set_xticks([], [])
    pca_ax.set_ylabel(f"PC2 ({adata.uns['pca']['variance_ratio'][1] * 100:.2f}%)", size=8)
    pca_ax.set_yticks([], [])

    pca_ax.annotate(
        f"t = {adata.obs[time_obs_key].iloc[center]:0.2f} min",
        (0.40, 0.05),
        xycoords='axes fraction',
        fontsize='medium',
        color='darkred'
    )
    
    gene_loc = adata.var_names.get_loc(gene)
    gene_data = expr_layer[:, gene_loc]
    
    try:
        gene_data = gene_data.A
    except AttributeError:
        pass

    delta_x = adata.obs[time_obs_key].iloc[connecteds] - adata.obs[time_obs_key].iloc[center]
    delta_y = gene_data[connecteds] - gene_data[center]

    lr = LinearRegression(fit_intercept=False).fit(delta_x.values.reshape(-1, 1), delta_y.reshape(-1, 1))
    slope_dxdt = lr.coef_[0][0]

    expr_ax.scatter(
        delta_x,
        delta_y,
        c=to_pool_colors(adata.obs['Pool'].iloc[connecteds]),
        s=1,
        alpha=0.75
    )

    expr_ax.scatter(
        0,
        0,
        c=to_pool_colors(adata.obs['Pool'].iloc[[center]]),
        s=20,
        edgecolors='r',
        linewidths=0.5,
    )

    if gene_name is None:
        gene_name = gene

    expr_ax.set_title(
        r"$\bf{" + gene_name + "}$: " + f"Cell {center}", size=8
    )
    expr_ax.set_xlabel("dt [min]", size=8)
    expr_ax.set_ylabel("dx [counts]", size=8)
    expr_ax.tick_params(labelsize=8)

    xlim = np.abs(delta_x).max()
    ylim = np.abs(delta_y).max()

    expr_ax.set_xlim(-1 * xlim, xlim)
    expr_ax.set_ylim(-1 * ylim, ylim)

    expr_ax.axline((0, 0), slope=slope_dxdt, linestyle='--', linewidth=1.0, c='black')
    expr_ax.annotate(
        f"{slope_dxdt:0.3f} [counts/min]",
        (0.20, 0.8),
        xycoords='axes fraction',
        fontsize='medium',
        color='black'
    )

    expr_ax.annotate(
        f"n = {len(connecteds)} cells",
        (0.40, 0.05),
        xycoords='axes fraction',
        fontsize='medium',
        color='darkred'
    )

    return fig
