import numpy as _np
import pandas as _pd
import pandas.api.types as _pat
import os as _os
import anndata as _ad
import scanpy as _sc
import scipy as _sp

from ..figure_constants import *
from .projection_common import *

class FigureSingleCellData:
    
    _all_data = None
    _expt_data = None
    
    expt_cats = [1, 2]
    gene_cats = ["WT", "fpr1"]       
        
    @property
    def expt_files(self):
        return {(e, g): RAPA_SINGLE_CELL_EXPR_BY_EXPT.format(e=e, g=g) 
                for e in self.expt_cats for g in self.gene_cats}
    
    @property
    def all_data(self):        
        return self._all_data
    
    @property
    def expt_data(self):    
        return self._expt_data
    
    @property
    def has_pca(self):
        return all(['X_pca' in self.all_data.obsm] +
                   ['X_pca' in v.obsm for k, v in self.expt_data.items()])
    
    @property
    def has_umap(self):
        return all(['X_umap' in self.all_data.obsm] +
                   ['X_umap' in v.obsm for k, v in self.expt_data.items()])
    
    @property
    def _all_adatas(self):
        return [self.all_data] + [v for k, v in self.expt_data.items()]
    
    def __init__(self):
        self._load()
    
    def do_projections(self):
        if not self.has_pca:
            self.apply_inplace_to_everything(do_pca, max(N_PCS))
        
        if not self.has_umap:
            self.apply_inplace_to_everything(do_umap, UMAP_NPCS, UMAP_NNS, UMAP_MIN_DIST)
            self.save()
    
    def apply_to_expts(self, func, *args, **kwargs):
        return [self._apply(x, func, *args, **kwargs) for _, x in self.expt_data.items()]
    
    def apply_to_everything(self, func, *args, **kwargs):            
        return [self._apply(x, func, *args, **kwargs) for x in self._all_adatas]    
    
    def apply_inplace_to_expts(self, func, *args, **kwargs):
        self.apply_to_expts(func, *args, **kwargs)
        
    def apply_inplace_to_everything(self, func, *args, **kwargs):            
        self.apply_to_everything(func, *args, **kwargs)
    
    @staticmethod
    def _apply(data, func, *args, **kwargs):
        if VERBOSE:
            _data_descript = str(data.obs["Experiment"].unique().astype(str)) + ", "
            _data_descript += str(data.obs["Gene"].unique().astype(str))
            print(f"Applying {func.__name__} to data [{_data_descript}] {data.shape}")
        return func(data, *args, **kwargs)
    
    def _load(self):
        _first_load = not _os.path.exists(RAPA_SINGLE_CELL_EXPR_PROCESSED)
        fn = RAPA_SINGLE_CELL_EXPR_PROCESSED if not _first_load else RAPA_SINGLE_CELL_EXPR_FILE
        
        if VERBOSE:
            print(f"Reading Single Cell Data from {fn}")
        
        self._all_data = _ad.read(fn)
        
        if _first_load:
            self._first_load(self._all_data)
            
        self._load_expts()
        
        if _first_load:
            self.save()
            
    
    def _load_expts(self):
        
        if self.expt_data is not None:
            return
        
        _expt_data = {}
        
        for k, v in self.expt_files.items():
            if _os.path.exists(v):
                if VERBOSE:
                    print(f"Reading Single Cell Experiment Data from {v}")
                _expt_data[k] = _ad.read(v)
            else:
                e, g = k
                if VERBOSE:
                    print(f"Extracting [{k}] Single Cell Data from all data")
                _expt_data[k] = self.all_data[(self.all_data.obs["Experiment"] == e) &
                                              (self.all_data.obs["Gene"] == g), :].copy()
        
        self._expt_data = _expt_data
        
    
    def save(self):
        
        # Save all data
        if self.all_data is not None:
            if VERBOSE:
                print(f"Writing Single Cell Data to {RAPA_SINGLE_CELL_EXPR_PROCESSED}")
            self._all_data.write(RAPA_SINGLE_CELL_EXPR_PROCESSED)
            
        # Save individual experiments
        if self.expt_data is not None:
            for k, v in self.expt_files.items():
                if VERBOSE:
                    print(f"Writing Single Cell Data to {v}")
                self.expt_data[k].write(v)
            
    
    @staticmethod
    def _first_load(adata):
        # Filter all-zero genes
        adata.raw = adata
        _sc.pp.filter_genes(adata, min_cells=10)

        # Copy counts and save basic count depth stats
        adata.layers['counts'] = adata.X.copy()
        adata.obs['n_counts'] = adata.X.sum(axis=1).astype(int)
        adata.obs['n_genes'] = adata.X.astype(bool).sum(axis=1)

        # Fix categorical dtypes
        if not _pat.is_categorical_dtype(adata.obs['Pool']):
            adata.obs['Pool'] = adata.obs['Pool'].astype("category")
            adata.obs['Experiment'] = adata.obs['Experiment'].astype("category")
            adata.obs['Gene'] = adata.obs['Gene'].astype("category")
            
        adata = _gene_metadata(adata)
        adata = _call_cc(adata)
        adata = calc_group_props(adata)
        adata = calc_other_cc_groups(adata)
        
        
def _gene_metadata(adata):
    # Gene common names
    if "CommonName" not in adata.var:
        if VERBOSE:
            print(f"Loading gene names from {GENE_NAMES_FILE}")
        yeast_gene_names = _pd.read_csv(GENE_NAMES_FILE, sep="\t", index_col=0)
        yeast_gene_names = yeast_gene_names.reindex(adata.var_names)
        na_names = _pd.isna(yeast_gene_names)
        yeast_gene_names[na_names] = adata.var_names.values[na_names.values.flatten()].tolist()
        adata.var["CommonName"] = yeast_gene_names.astype(str)

    # Gene groups
    if "iESR" not in adata.var:
        if VERBOSE:
            print(f"Loading gene metadata from {GENE_GROUP_FILE}")
        ygg = _pd.read_csv(GENE_GROUP_FILE, sep="\t", index_col=0)
        ygg = (ygg.pivot(columns="Group", values="Source").fillna(0) != 0).reindex(adata.var_names).fillna(False)
        adata.var = _pd.concat((adata.var, ygg.astype(bool)), axis=1)
        adata.var.rename({"M/G1": "M-G1"}, inplace=True, axis=1)
        
    return adata


def _call_cc(data):
    
    if "CC" not in data.obs:
        if VERBOSE:
            print(f"Assigning Cell Cycle {CC_COLS}")
        # Generate a proportion table for each of the cell cycle categories
        cc_prop = _np.zeros((data.shape[0], len(CC_COLS)), order="F")
        for i, cc in enumerate(CC_COLS):
            cc_prop[:, i] = data.layers['counts'][:, data.var[cc].values].sum(axis=1).A1
            cc_prop[:, i] /= data.obs['n_counts']

        # Z-score each of the columns of the table (within each category)
        cc_prop = _sp.stats.zscore(cc_prop, ddof=1)
        
        # Take the largest Z-scored value for each cell
        data.obs["CC"] = _np.array(CC_COLS)[_np.argmax(cc_prop, axis=1)]
        cat_type = _pat.CategoricalDtype(categories=CC_COLS, ordered=True)
        data.obs["CC"] = data.obs["CC"].astype(cat_type)
        
    return data

def _run_dewakss


def calc_group_props(data, cols=AGG_COLS):
    
    if cols[0] not in data.obs:
        if VERBOSE:
            print(f"Calculating aggregate proportion {cols}")
        for i, ag in enumerate(cols):
            ag_ratio = data.layers['counts'][:, data.var[ag].values].sum(axis=1).A1.astype(float)
            ag_ratio /= data.obs['n_counts']
            data.obs[ag] = ag_ratio
            
    return data

def calc_other_cc_groups(data):
    
    data.var[CELLCYCLE_GROUP_COL] = data.var[CC_COLS].any(axis=1)
    data.var[OTHER_GROUP_COL] = ~data.var[CC_COLS + AGG_COLS].any(axis=1)
    
    calc_group_props(data, [CELLCYCLE_GROUP_COL, OTHER_GROUP_COL])
    
    return data


# Load and return RAPAMYCIN bulk RNA-seq expression data
def load_rapa_bulk_data():
    if VERBOSE:
        print(f"Loading Bulk Data from {RAPA_BULK_EXPR_FILE}")
    rapa_bulk_data = _pd.read_csv(RAPA_BULK_EXPR_FILE, sep="\t", index_col=0)
    rapa_bulk_metadata = rapa_bulk_data[RAPA_BULK_EXPR_FILE_META_DATA_COLS].astype(str).astype("category")
    rapa_bulk_data.drop(RAPA_BULK_EXPR_FILE_META_DATA_COLS, axis=1, inplace=True)
    return rapa_bulk_data, rapa_bulk_metadata

def rapa_bulk_times(include_0=False):
    rbt = RAPA_BULK_EXPR_FILE_TIMES if not include_0 else [0] + RAPA_BULK_EXPR_FILE_TIMES
    return _pd.Series(rbt).astype(str).tolist()

def sum_for_pseudobulk(adata, by_cols):

    group_bulk = []
    meta_bulk = []
    
    grouper = adata.obs.groupby(by_cols)
    for groups, group_data in grouper:
        idx = adata.obs_names.isin(group_data.index)
        group_bulk.append(_pd.DataFrame(_np.sum(adata.X[idx, :], axis=0).reshape(1, -1), 
                                        columns=adata.var_names))
        meta_bulk.append(_pd.DataFrame([groups], columns=by_cols))

    group_bulk = _pd.concat(group_bulk)
    group_bulk.reset_index(drop=True, inplace=True)
    
    meta_bulk = _pd.concat(meta_bulk).astype('str').astype("category")
    meta_bulk.reset_index(drop=True, inplace=True)
    
    return group_bulk, meta_bulk
