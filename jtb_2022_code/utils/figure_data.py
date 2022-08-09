import itertools
import functools

import numpy as _np
import pandas as _pd
import pandas.api.types as _pat
import os as _os
import gc as _gc
import anndata as _ad
import scanpy as _sc
import scipy as _sp
import inferelator_velocity as _ifv

from ..figure_constants import *
from .projection_common import *
from .dewakss_common import run_dewakss
from .decay_common import calc_decays, calc_decay_windows, _calc_decay_windowed, _calc_decay
from .velocity_common import calculate_velocities
from .pseudotime_common import spearman_rho_grid, calc_rhos, spearman_rho_pools
from .process_published_data import process_all_decay_links
from sklearn.metrics import pairwise_distances

class FigureSingleCellData:
    
    _all_data = None
    _expt_data = None
    _expt_keys = None
       
    expt_cats = [1, 2]
    gene_cats = ["WT", "fpr1"]   
        
    @property
    def expt_files(self):
        return {(e, g): RAPA_SINGLE_CELL_EXPR_BY_EXPT.format(e=e, g=g) 
                for e in self.expt_cats for g in self.gene_cats}
    
    @property
    def all_data_file(self):
        return RAPA_SINGLE_CELL_EXPR_PROCESSED
    
    @property
    def all_data(self):        
        return self._all_data
    
    @property
    def expt_data(self):    
        return self._expt_data
    
    @property
    def expts(self):
        return self._expt_keys            
    
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
       
    def __init__(self, start_from_scratch=False, memory_efficient=True):
        self._load(from_unprocessed=start_from_scratch)
           
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
        if VERBOSE > 1:
            _data_descript = str(data.obs["Experiment"].unique().astype(str)) + ", "
            _data_descript += str(data.obs["Gene"].unique().astype(str))
            print(f"Applying {func.__name__} to data [{_data_descript}] {data.shape}")
        return func(data, *args, **kwargs)
    
    def _load(self, from_unprocessed=False):
        _first_load = from_unprocessed or not _os.path.exists(RAPA_SINGLE_CELL_EXPR_PROCESSED)
        fn = RAPA_SINGLE_CELL_EXPR_PROCESSED if not _first_load else RAPA_SINGLE_CELL_EXPR_FILE
        
        if VERBOSE:
            print(f"Reading Single Cell Data from {fn}")
        
        self._all_data = _ad.read(fn)
        
        if _first_load:
            self._first_load(self._all_data)
            
        self._load_expts(force_extraction_from_all=from_unprocessed)
        
        if _first_load:
            self.apply_inplace_to_everything(self._normalize)
            self.save()
            
    
    def _load_expts(self, force_extraction_from_all=False):
        
        if self.expt_data is not None:
            return
        
        self._expt_keys = [k for k in self.expt_files]
        self._expt_data = {k: None for k in self._expt_keys}
        
        for k, v in self.expt_files.items():
            if _os.path.exists(v) and not force_extraction_from_all:
                if VERBOSE:
                    print(f"Reading Single Cell Experiment Data from {v}")
                self._expt_data[k] = _ad.read(v)
                
            else:
                e, g = k
                if VERBOSE:
                    print(f"Extracting [{k}] Single Cell Data from all data")
                    
                self._expt_data[k] = self.all_data[(self.all_data.obs["Experiment"] == e) &
                                                   (self.all_data.obs["Gene"] == g), :].copy()
    
    @staticmethod
    def _normalize(adata):
        
        adata.layers['counts'] = adata.X.copy()
        adata.X = adata.X.astype(float)
        _sc.pp.normalize_per_cell(adata)
        _sc.pp.log1p(adata)
        
        return adata
                
    
    def save(self):
        
        # Save all data
        if self.all_data is not None:
            if VERBOSE:
                print(f"Writing Single Cell Data to {RAPA_SINGLE_CELL_EXPR_PROCESSED}")
            self._all_data.write(RAPA_SINGLE_CELL_EXPR_PROCESSED)
            
        # Save individual experiments
        if self.expt_data is not None:
            for k, v in self.expt_files.items():
                if self.expt_data[k] is not None:
                    if VERBOSE:
                        print(f"Writing Single Cell Data to {v}")
                    self.expt_data[k].write(v)
            
    
    def load_pseudotime(self, files=PSEUDOTIME_FILES, do_rho=True):
                
        ### LOAD FLATFILES ###
        for k, (fn, has_idx) in files.items():
            pt_key = self._pseudotime_key(k[0], k[1])
            if pt_key not in self.all_data.obsm or any(pt_key not in x.obsm for _, x in self.expt_data.items()):
                loaded = _pd.read_csv(fn, sep="\t", index_col=0 if has_idx else None)
                loaded.index = loaded.index.astype(str)
                self.all_data.obsm[pt_key] = loaded

                for (expt, gene), expt_v in self.expt_data.items():
                    expt_idx = self.all_data.obs['Experiment'] == expt
                    expt_idx &= self.all_data.obs['Gene'] == gene
                    expt_v.obsm[pt_key] = self.all_data.obsm[pt_key].loc[expt_idx, :].copy()
                    
                if k[0] == 'palantir' and not k[1]:
                    self.apply_inplace_to_everything(_fix_palantir, pt_key)

        ### CALCULATE CORRELATIONS AGAINST THE TIME DATA ###
        def _get_rho(adata, key):
            return spearman_rho_pools(adata.obs['Pool'], adata.obs[key])
        
        if do_rho:
                    
            if 'rho' not in self.all_data.uns:
                self.all_data.uns['rho'] = _pd.DataFrame(index=_pd.MultiIndex.from_tuples(self.expts))

            if 'pca_pt' in self.expt_data[(1, "WT")].obs:
                df = _pd.DataFrame(self.apply_to_expts(_get_rho, 'pca_pt'), 
                                   index=_pd.MultiIndex.from_tuples(self.expts),
                                   columns = ['pca'])
                self.all_data.uns['rho'] = _pd.concat((self.all_data.uns['rho'], df), axis=1)


            if 'denoised_rho' not in self.all_data.uns:
                self.all_data.uns['denoised_rho'] = _pd.DataFrame(index=_pd.MultiIndex.from_tuples(self.expts))

            if 'denoised_pca_pt' in self.expt_data[(1, "WT")].obs:
                df = _pd.DataFrame(self.apply_to_expts(_get_rho, 'denoised_pca_pt'), 
                                   index=_pd.MultiIndex.from_tuples(self.expts),
                                   columns = ['pca'])
                self.all_data.uns['denoised_rho'] = _pd.concat((self.all_data.uns['denoised_rho'], df), axis=1)

            ### CALCULATE SPEARMAN RHO FOR EACH EXPERIMENT ###
            for k, _ in files.items():
                rho_key = k[0] + '_rho'
                pt_key = self._pseudotime_key(k[0], k[1])

                if rho_key not in self.all_data.uns and not k[1]:
                    spearman_rho_grid(self, pt_key, rho_key)
                    rhomax = _pd.DataFrame(_np.abs(self.all_data.uns[rho_key]).apply(_np.nanmax, axis=1))
                    rhomax.columns = [k[0]]
                    self.all_data.uns['rho'] = _pd.concat((self.all_data.uns['rho'], rhomax), axis=1)

                if k[0] not in self.all_data.uns['denoised_rho'].columns and k[1]:
                    df = _pd.DataFrame(self.apply_to_expts(calc_rhos, pt_key))
                    df.columns = [k[0]]
                    df.index = _pd.MultiIndex.from_tuples(self.expts)
                    df = df.applymap(lambda x: x[1])
                    self.all_data.uns['denoised_rho'] = _pd.concat((self.all_data.uns['denoised_rho'], df), axis=1)

            self.all_data.uns['denoised_rho'] = _np.abs(self.all_data.uns['denoised_rho'])

        ### CALCULATE TIME VALUES BASED ON PSEUDOTIME ###
        
        def _pt_to_obs(adata):
            for m in ['dpt', 'cellrank', 'monocle', 'palantir']:
                km = m + '_pt'
                adata.obs[km] = adata.obsm[self._pseudotime_key(m, True)].copy()
        
        self.apply_inplace_to_expts(_pt_to_obs)
        
    def get_pseudotime(method, denoised=False, expt=None):
        if expt is None:
            return self.all_data.obsm[self._pseudotime_key(method, denoised)]
        else:
            return self.expt_data[expt].obsm[self._pseudotime_key(method, denoised)]
        
    def load_published_decay(self):
        
        if not all(x in self.all_data.var.columns for x in DECAY_CONSTANT_FILES.keys()):        
            p_decay = process_all_decay_links(self.all_data.var_names)
            self.all_data.var[p_decay.columns] = p_decay
            
    def velocity_data(self, expt, gene):
        
        _fn = RAPA_SINGLE_CELL_VELOCITY_BY_EXPT.format(e=expt, g=gene)
        
        if _os.path.exists(_fn):
            print(f"Loading velocity data from {_fn}")
            return _ad.read(_fn)
        
        else:
            print(f"{_fn} not found. Generating velocity data:")
            adata = self.denoised_data(expt, gene)
            
            adata.obs[RAPA_TIME_COL] = self.expt_data[(expt, gene)].obs[RAPA_TIME_COL]
            adata.obs[CC_TIME_COL] = self.expt_data[(expt, gene)].obs[CC_TIME_COL]
                       
            calculate_velocities(adata)
            adata.write(_fn)
            
            return adata
        
    def decay_data(self, expt, gene, recalculate=False):
        
        _velo_data = self.velocity_data(expt, gene)
        
        if 'cell_cycle_decay' not in _velo_data.var.columns or recalculate:
            print("Calculating Biophysical Paramaters:")
            
            for g_key, time_key, tmin, tmax in [('cell_cycle', CC_TIME_COL, 0, 88), ('rapamycin', RAPA_TIME_COL, -10, 60)]:

                calc_decays(
                    _velo_data,
                    g_key + "_velocity",
                    output_key = g_key + "_decay",
                    output_alpha_key = g_key + "_alpha",
                    force=recalculate
                )

                calc_decay_windows(_velo_data,
                    g_key + "_velocity",
                    time_key,
                    output_key = g_key + "_window_decay",
                    output_alpha_key = g_key + "_window_alpha",
                    include_alpha=True,
                    t_min=tmin,
                    t_max=tmax,
                    force=recalculate
                )

            _fn = RAPA_SINGLE_CELL_VELOCITY_BY_EXPT.format(e=expt, g=gene)
            
            if VERBOSE:
                print(f"Writing Single Cell Decay Data to {_fn}")
                
            _velo_data.write(_fn)
            
        return _velo_data
    
    def decay_data_all(self, recalculate=False):
        
        if recalculate or 'cell_cycle_decay' not in self.all_data.var.columns:
        
            adata_list = [self.decay_data(1, "WT"), self.decay_data(2, "WT")]

            for g_key, time_key, tmin, tmax in [('cell_cycle', CC_TIME_COL, 0, 88), ('rapamycin', RAPA_TIME_COL, -10, 60)]:
                expr = _np.vstack([a.X for a in adata_list])
                velo = _np.vstack([a.layers[g_key + "_velocity"] for a in adata_list])
                times = _np.hstack([a.obs[time_key].values for a in adata_list])
                
                decays, decays_se, a = _calc_decay(
                    expr,
                    velo,
                    include_alpha=True
                )
                
                self.all_data.var[g_key + "_decay"] = decays
                self.all_data.var[g_key + "_decay_se"] = decays_se
                self.all_data.var[g_key + "_alpha"] = a
                
                                
                self.all_data.uns[g_key + "_decay"] = {'params': 
                                                    {'include_alpha': True, 
                                                     'decay_quantiles': [0.0, 0.05],
                                                     'bootstrap': False}}
                                
                decays, decays_se, a, t_c = _calc_decay_windowed(
                    expr,
                    velo,
                    times,
                    t_min=tmin,
                    t_max=tmax,
                    include_alpha=True,
                    bootstrap=False,
                    time_wrap=None if time_key != CC_TIME_COL else CC_LENGTH
                )
                
                self.all_data.uns[g_key + "_window_decay"] = {'params': 
                                                    {'include_alpha': True, 
                                                     'decay_quantiles': [0.0, 0.05],
                                                     'bootstrap': False},
                                                    'times': t_c}
                
                self.all_data.varm[g_key + "_window_decay"] = _np.array(decays).T
                self.all_data.varm[g_key + "_window_decay_se"] = _np.array(decays_se).T
                self.all_data.varm[g_key + "_window_alpha"] = _np.array(a).T
                
            
            del adata_list
            _gc.collect()
                
            if VERBOSE:
                print(f"Writing Single Cell Data to {RAPA_SINGLE_CELL_EXPR_PROCESSED}")
                    
            self._all_data.write(RAPA_SINGLE_CELL_EXPR_PROCESSED)
                
        return self.all_data
            
    def denoised_data(self, expt, gene):
        
        _fn = RAPA_SINGLE_CELL_DENOISED_BY_EXPT.format(e=expt, g=gene)
        
        if _os.path.exists(_fn):
            print(f"Loading denoised data from {_fn}")
            return _ad.read(_fn)
        
        else:
            print(f"{_fn} not found. Generating denoised data:")
            adata = _ad.AnnData(self.expt_data[(expt, gene)].layers['counts'].copy(),
                                obs=self.expt_data[(expt, gene)].obs.copy(),
                                var=self.expt_data[(expt, gene)].var.copy(),
                                dtype=int)
            
            if 'noise2self_distance_graph' not in self.expt_data[(expt, gene)].obsp:
                _ifv.global_graph(self.expt_data[(expt, gene)], verbose=True)
                
            adata.obsp['noise2self_distance_graph'] = self.expt_data[(expt, gene)].obsp['noise2self_distance_graph'].copy()
            run_dewakss(adata)            
            adata.write(_fn)
            
            return adata           
            
    @staticmethod
    def _pseudotime_key(method, denoised=False):
        return str(method).lower() + "_" + str(denoised)
    
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
        
        _add_broad_category(adata)
        
    def gene_common_name(self, gene_symbol):
        
        if gene_symbol in self._all_data.var_names and "CommonName" in self._all_data.var.columns:
            return self._all_data.var.loc[gene_symbol, "CommonName"]
        
        else:
            return None
        
    def process_programs(self, recalculate=False):
        
        if recalculate or 'program' not in self.all_data.var:
            _ifv.program_select(
                self.all_data,
                layer='counts',
                verbose=True
            )
                
            if _np.sum(self.all_data.var['program'] == '0') > _np.sum(self.all_data.var['program'] == '1'):
                _rapa_program, _cc_program = '0', '1'
            else:
                _rapa_program, _cc_program = '1', '0'

            self.all_data.uns['programs']['rapa_program'] = _rapa_program
            self.all_data.uns['programs']['cell_cycle_program'] = _cc_program

            def _transfer_programs(adata):
                adata.var['program'] = self.all_data.var['program'].reindex(adata.var_names)
                adata.obs['Pool_Combined'] = adata.obs['Pool'].astype(str)
                adata.obs.loc[adata.obs['Pool_Combined'] == '1', 'Pool_Combined'] = '12'
                adata.obs.loc[adata.obs['Pool_Combined'] == '2', 'Pool_Combined'] = '12'

            self.apply_inplace_to_expts(
                _transfer_programs
            )
            
            self.apply_inplace_to_expts(
                _ifv.times.program_times,
                {_rapa_program: 'Pool_Combined',
                 _cc_program: 'CC'},
                {_rapa_program: RAPA_TIME_ORDER,
                 _cc_program: CC_TIME_ORDER},
                layer='counts',
                verbose=True
            )
            
            self.apply_inplace_to_expts(
                _ifv.global_graph,
                verbose=True
            )

            def _sort_out_times(adata):
                adata.obs[RAPA_TIME_COL] = adata.obs[f'program_{_rapa_program}_time'].copy()
                adata.obs[CC_TIME_COL] = adata.obs[f'program_{_cc_program}_time'].copy()
                
            self.apply_inplace_to_expts(_sort_out_times)
            
            self.save()
            
    def calc_gene_dists(self, recalculate=False):

        if recalculate or 'cosine_distance' not in self.all_data.uns['programs']:

            _dists = _ad.AnnData(
                self.all_data.layers['counts'],
                dtype=float,
                var=self.all_data.var
            )

            _sc.pp.normalize_per_cell(_dists)
            _sc.pp.log1p(_dists)   
            _sc.pp.highly_variable_genes(_dists, max_mean=_np.inf, min_disp=0.01)

            _dists._inplace_subset_var(_dists.var['highly_variable'].values)

            _sc.pp.pca(_dists, n_comps=self.all_data.uns['programs']['n_comps'])    

            print(f"Using {self.all_data.uns['programs']['n_comps']} PCs for distance")
            _dists = (_dists.obsm['X_pca'] @ _dists.varm['PCs'].T).T

            _gc.collect()

            for metric in ['cosine', 'euclidean', 'manhattan']:
                print(f"Calculating metric {metric} on data {_dists.shape}")
                self.all_data.uns['programs'][f'{metric}_distance'] = pairwise_distances(
                    _dists, metric=metric
                )

            del _dists

            self.save()


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

### SELECT 15 DCs FROM PALANTIR: ###
def _dc_select(obsm_data, n_dcs=15):
    col_split = list(map(lambda x: x.split("_"), obsm_data.columns))
    keep_cols = [int(x[1]) == n_dcs for x in col_split]
    obsm_data = obsm_data.loc[:, keep_cols].copy()
    obsm_data.columns = [str(x[0]) + "_" + str(x[2]) 
                         for x, y in zip(col_split, keep_cols) if y]
    return obsm_data

def _fix_palantir(adata, obsm_key):
    nd = _dc_select(adata.obsm[obsm_key])
    adata.obsm[obsm_key] = nd

def _add_broad_category(adata):
    adata.var['category'] = "NA"
    adata.var.loc[functools.reduce(lambda x, y: x | adata.var[y], [adata.var['G1'], 'G2', 'M', 'M-G1', 'S']), 'category'] = "CC"
    adata.var.loc[functools.reduce(lambda x, y: adata.var[x] | adata.var[y], ['RP', 'RiBi']), 'category'] = "RP"
    adata.var['category'] = _pd.Categorical(adata.var['category'], categories=["NA","RP","CC"])
