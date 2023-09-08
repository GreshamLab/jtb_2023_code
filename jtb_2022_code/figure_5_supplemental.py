import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from jtb_2022_code.utils.figure_common import *
from jtb_2022_code.utils.process_published_data import process_all_decay_links

from jtb_2022_code.figure_constants import *
from .figure_3 import _get_fig4_data, _fig3_plot

from inferelator_velocity.decay import calc_decay
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr


def figure_5_supplement_1_plot(data, f3_data=None, save=True):
    
    if f3_data is None:
        f3_data = _get_fig4_data(data)
        
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
        fig.savefig(FIGURE_5_SUPPLEMENTAL_FILE_NAME + "_1.png", facecolor='white')
        
    return fig


def figure_5_supplement_2_plot(data, save=True):

    pubbed = process_all_decay_links(data.all_data.var_names)
    pubbed['Jackson2023'] = np.log(2) / data.all_data.varm['rapamycin_window_decay'][:, 1:9].mean(1)
    pubbed = pubbed.dropna()
    
    order = ['Jackson2023', 'Neymotin2014', 'Chan2018', 'Miller2011', 'Munchel2011', 'Geisberg2014']
    
    fig, axd = plt.subplots(6, 6, figsize=(6, 6), dpi=MAIN_FIGURE_DPI)
    
    ax_hm = fig.add_axes([0.65, 0.65, 0.2, 0.2])
    ax_hm_cbar = fig.add_axes([0.875, 0.7, 0.02, 0.1])
    
    corr_mat = np.full((len(order), len(order)), np.nan, dtype=float)
    
    for i, dataset in enumerate(order):
        for j, dataset2 in enumerate(order[i:]):
            corr_mat[i + j, i] = spearmanr(
                pubbed[dataset2],
                pubbed[dataset]
            ).statistic
                   
    hm_ref = ax_hm.pcolormesh(
        corr_mat[::-1, :],
        cmap='Reds',
        vmin=0,
        vmax=1
    )

    for i in range(len(order)):
        for j in range(len(order) - i):
            ax_hm.text(j + 0.5, i + 0.5, f"{corr_mat[::-1][i, j]:.2f}", 
                horizontalalignment='center', 
                verticalalignment='center',
                size=4
            )
    
    ax_hm.set_xticks(
        np.arange(len(order)) + 0.5,
        order,
        rotation=90,
        size=6
    )
    
    ax_hm.set_yticks(
        np.arange(len(order)) + 0.5,
        order[::-1],
        size=6
    )

    axd[0, 0].set_title("A", loc='left', weight='bold', size=8, y=0.85, x=-0.4)
    ax_hm.set_title("B", loc='left', weight='bold', size=8)
    
    plt.colorbar(hm_ref, ax_hm_cbar)
    ax_hm_cbar.set_yticks([0, 1], [0, 1], size=8)
    ax_hm_cbar.tick_params(axis='y', length=2, pad=1)
    ax_hm_cbar.set_title("ρ", size=8)

    for i, dataset in enumerate(order):
        axd[i, 0].set_ylabel(dataset, size=6)
        axd[-1, i].set_xlabel(dataset, size=6)

        for j, dataset2 in enumerate(order[i:]):
            axd[j + i, i].scatter(
                pubbed[dataset2],
                pubbed[dataset],
                s=1,
                alpha=0.1,
                color='black'
            )
            axd[j + i, i].set_xticks([], [])
            axd[j + i, i].set_yticks([], [])
            axd[j + i, i].set_xlim(0, np.quantile(pubbed[dataset2], 0.99))
            axd[j + i, i].set_ylim(0, np.quantile(pubbed[dataset], 0.99))
            model = LinearRegression(fit_intercept=False).fit(
                pubbed[dataset2].values.reshape(-1, 1),
                pubbed[dataset].values.reshape(-1, 1)
            )
            r2 = corr_mat[i + j, i]
            axd[j + i, i].axline((0, 0), slope=model.coef_, color='red', linestyle="--", linewidth=1)
            axd[j + i, i].annotate(
                f"ρ={r2:.2f}",
                (0.1, 0.75),
                xycoords='axes fraction',
                color='black',
                size=6,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='white', boxstyle='round')
            )

        for k in range(i):
            axd[k, i].axis('off')

    if save:
        fig.savefig(FIGURE_5_SUPPLEMENTAL_FILE_NAME + "_2.png", facecolor='white')
        
    return fig
