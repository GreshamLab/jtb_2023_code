import numpy as _np
import pandas as _pd
import pandas.api.types as _pat
import os as _os
import anndata as _ad
import scanpy as _sc
import scipy as _sp
import matplotlib.colors as _colors
import matplotlib.cm as _cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from seaborn.matrix import dendrogram as _sns_dendrogram

from ..figure_constants import *


def _cmap_to_palette(cmap, n):
    cmap = _cm.get_cmap(cmap, n)
    return [_colors.rgb2hex(cmap(x)) for x in range(n)]
       
def pool_palette():
    return _cmap_to_palette(POOL_PALETTE, 9)[1:9]

def to_pool_colors(series_data):
    return series_data.map({k: v for k, v in zip(range(1, 9), pool_palette())}).values

def expt_palette(long=False):
    ep = _cmap_to_palette(EXPT_PALETTE, 8)
    return [ep[2], ep[5]] if not long else [ep[2], ep[5], ep[1], ep[6]]

def strain_palette():
    ep = _cmap_to_palette(GENE_PALETTE, 8)
    return [ep[0], ep[3]]

def cc_palette():
    return _cmap_to_palette(CC_PALETTE, len(CC_COLS))

def to_cc_colors(series_data):
    return series_data.map({k: v for k, v in zip(CC_COLS, cc_palette())}).values

def gene_category_palette():
    return _cmap_to_palette(GENE_CAT_PALETTE, len(GENE_CAT_COLS))

def squeeze_data(data, high, low=None):
    low = -1 * high if low is None else low
    data = data.copy()
    data[data > high] = high
    data[data < low] = low
    return data

def add_legend_axis(ax, size='8%', pad=0.05, add_extra_pad=None):
    divider = make_axes_locatable(ax)
    lax = divider.append_axes('right', size=size, pad=pad)
    lax.axis('off')
    
    if add_extra_pad is not None:
        divider.append_axes('right', size=size, pad=pad)
    
    return lax

def add_legend(ax, colors, labels, title=None, horizontal=False, **kwargs):
    ax.axis('off')
    _ = [ax.scatter([], [], c=c, label=l) for c, l in zip(colors, labels)]
    return ax.legend(frameon=False,
                     loc='center left',
                     ncol=len(labels) if horizontal else 1,
                     handletextpad=0.1 if horizontal else 0.8,
                     borderpad=0.1,
                     borderaxespad=0.1,
                     columnspacing=1 if horizontal else 0,
                     mode=None,
                     title=title,
                     **kwargs)


def add_legend_in_plot(ax, colors, labels, title=None, frameon=True):
    fakeplots = [ax.scatter([], [], c=c, label=l) for c, l in zip(colors, labels)]
    return ax.legend(frameon=frameon, 
                     loc='center right',
                     bbox_to_anchor=(0, 0.85),
                     ncol=1,
                     columnspacing=0,
                     mode=None,
                     title=title)

def plot_heatmap(figure, matrix_data, matrix_cmap, matrix_ax, dendro_data=None, dendro_linkage=None, dendro_ax=None, 
                 row_data=None, row_cmap=None, row_ax=None, row_xlabels=None, vmin=None, vmax=None,
                 add_colorbar=True, colorbar_ax=None, colorbar_label=None, colorbar_loc=None):
    
    refs = {}
    
    refs['matrix'] = matrix_ax.pcolormesh(matrix_data, cmap=matrix_cmap, vmin=vmin, vmax=vmax)
    matrix_ax.invert_yaxis()
    matrix_ax.set_xticks([])
    matrix_ax.set_yticks([])
    matrix_ax.spines['right'].set_visible(False)
    matrix_ax.spines['top'].set_visible(False)
    matrix_ax.spines['left'].set_visible(False)
    matrix_ax.spines['bottom'].set_visible(False)

    if row_data is not None:
        _n_row_cols = row_data.shape[1] if row_data.ndim > 1 else 1
        
        refs['rows'] = row_ax.pcolormesh(row_data, cmap=row_cmap)
        row_ax.invert_yaxis()
        row_ax.set_yticks([])
        row_ax.spines['right'].set_visible(False)
        row_ax.spines['top'].set_visible(False)
        row_ax.spines['left'].set_visible(False)
        row_ax.set_xticks(_np.arange(_n_row_cols) + 0.5)
        
    if row_xlabels is not None:
        row_ax.set_xticklabels(row_xlabels, rotation=90, va='top', fontsize=6)
    
    if dendro_ax is not None:
        refs['dendro'] = _sns_dendrogram(dendro_data, metric='correlation', method='average', label=False,
                                         axis=0, rotate=True, ax=dendro_ax, 
                                         linkage=dendro_linkage, tree_kws={'linewidths': 0.05})
        dendro_ax.invert_xaxis()
        dendro_ax.axis('off')
    
    if add_colorbar:
        vmin = np.nanmin(matrix_data) if vmin is None else vmin
        vmax = np.nanmax(matrix_data) if vmax is None else vmax
        refs['cax'] = figure.add_axes(colorbar_loc) if colorbar_ax is None else colorbar_ax
        refs['cbar'] = figure.colorbar(refs['matrix'], cax=refs['cax'], orientation='vertical', ticks=[vmin, vmax])
        refs['cax'].yaxis.set_tick_params(pad=0)
        if vmax >= 100000:
            refs['cax'].set_yticklabels([str(vmin), f'{int(vmax / 1000)}k'])
        elif vmax >= 1000:
            refs['cax'].set_yticklabels([str(vmin), f'{vmax / 1000:.1f}k'])
            
    if colorbar_label is not None:
        refs['cbar'].set_label(colorbar_label, labelpad=-1)
        
    return refs
