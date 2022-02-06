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

from ..figure_constants import *


def _cmap_to_palette(cmap, n):
    cmap = _cm.get_cmap(cmap, n)
    return [_colors.rgb2hex(cmap(x)) for x in range(n)]
       
def pool_palette():
    return _cmap_to_palette(POOL_PALETTE, 9)[1:9]

def expt_palette(long=False):
    ep = _cmap_to_palette(EXPT_PALETTE, 8)
    return [ep[2], ep[5]] if not long else [ep[2], ep[5], ep[1], ep[6]]

def strain_palette():
    ep = _cmap_to_palette(GENE_PALETTE, 8)
    return [ep[0], ep[3]]

def cc_palette():
    return _cmap_to_palette(CC_PALETTE, len(CC_COLS))

def gene_category_palette():
    return _cmap_to_palette(GENE_CAT_PALETTE, len(GENE_CAT_COLS))

def squeeze_data(data, high, low=None):
    low = -1 * high if low is None else low
    data = data.copy()
    data[data > high] = high
    data[data < low] = low
    return data

def add_legend_axis(ax, size='8%', pad=0.05):
    divider = make_axes_locatable(ax)
    lax = divider.append_axes('right', size=size, pad=pad)
    lax.axis('off')
    return lax

def add_legend(ax, colors, labels, title=None):
    fakeplots = [ax.scatter([], [], c=c, label=l) for c, l in zip(colors, labels)]
    return ax.legend(frameon=False, 
                     loc='center left', 
                     ncol=1,
                     borderpad=0.1, 
                     borderaxespad=0.1,
                     columnspacing=0,
                     mode=None,
                     title=title)
