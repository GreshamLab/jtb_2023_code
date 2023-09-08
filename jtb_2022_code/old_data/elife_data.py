import os
import pandas as pd
import anndata as ad
import numpy as np
import scipy.sparse
import scanpy as sc

import inferelator_velocity as ifv

from ..figure_constants import *
from ..utils.figure_data import _gene_metadata, _call_cc
from ..utils.dewakss_common import run_dewakss

TF_LOOKUP = {
    "WT(ho)": None,
    "gln3": "YER040W",
    "gat1": "YFL021W",
    "gzf3": "YJL110C",
    "rtg1": "YOL067C",
    "rtg3": "YBL103C",
    "stp1": "YDR463W",
    "stp3": "YHR006W",
    "dal80": "YKR034W",
    "dal81": "YIR023W",
    "dal82": "YNL314W",
    "gcn4": "YEL009C"
}

class OldElifeData:
    
    data = None
    pseudobulk = None
    
    def __init__(self, align_adata=None, force_reload=False):
        
        if force_reload:
            self._first_load(align_adata=align_adata)
        
        elif os.path.exists(ELIFE_SINGLE_CELL_FILE_PROCESSED):
            print(f"Loading {ELIFE_SINGLE_CELL_FILE_PROCESSED}")
            self.data = ad.read(ELIFE_SINGLE_CELL_FILE_PROCESSED)
            
        else:
            self._first_load(align_adata=align_adata)
            
    def _first_load(self, align_adata=None):
        
        print(f"Loading and processing {ELIFE_SINGLE_CELL_FILE}")
        
        df = ad.read(ELIFE_SINGLE_CELL_FILE)

        obs_data = df.obs.copy()
        df = df.to_df()
        
        if align_adata is not None:
            
            df = df.reindex(
                align_adata.var_names,
                axis=1
            ).fillna(
                0.
            ).astype(
                np.float32
            )
                    
        self.data = ad.AnnData(
            df,
            obs=obs_data,
            dtype=np.float32
        )
        
        self.data.layers['counts'] = scipy.sparse.csr_matrix(
            self.data.X.astype(np.int32)
        )
        
        if align_adata is not None and 'programs' in align_adata.var:
            self.data.var['programs'] = align_adata.var['programs']
            
        if align_adata is not None and 'programs' in align_adata.uns:
            self.data.uns['programs'] = align_adata.uns['programs']
            
        self.data.obs['n_counts'] = self.data.X.sum(axis=1).astype(int)
        self.data.obs['n_genes'] = self.data.X.astype(bool).sum(axis=1)
        
        self.data = _gene_metadata(self.data)
        self.data = _call_cc(self.data)

        print(f"Writing {ELIFE_SINGLE_CELL_FILE_PROCESSED}")
        self.data.write(ELIFE_SINGLE_CELL_FILE_PROCESSED)
        
    def assign_times(self, force=False):
        
        if 'cell_cycle_time' not in self.data.obs or force:
            self.data.obs['cell_cycle_time'] = pd.NA

            for cond in self.data.obs['Condition'].dtype.categories.values:

                print(f"Calculating cell cycle times for {cond}")

                adata = self.get_data(genotype=None, condition=cond)
                _idx = self._get_index(None, cond)

                ifv.times.program_times(
                    adata,
                    {'1': 'CC'},
                    {'1': CC_TIME_ORDER},
                    layer='counts',
                    programs='1',
                    verbose=True
                )

                ifv.times.wrap_times(
                    adata,
                    '1',
                    CC_LENGTH
                )

                self.data.uns[f'{cond}_program_1'] = adata.uns['program_1_pca']
                self.data.obs.loc[_idx, 'cell_cycle_time'] = adata.obs['program_1_time']
            
            self.data.obs['cell_cycle_time'] = self.data.obs['cell_cycle_time'].astype(float)
            
            print(f"Writing {ELIFE_SINGLE_CELL_FILE_PROCESSED}")
            self.data.write(ELIFE_SINGLE_CELL_FILE_PROCESSED)
            
    def denoise(self, force=False):
        
         if 'denoised' not in self.data.layers or force:

            self.data.layers['denoised'] = np.full_like(self.data.X, np.nan)

            for cond in self.data.obs['Condition'].dtype.categories.values:

                print(f"Denoising {cond}")

                adata = self.get_data(genotype=None, condition=cond)
                _idx = self._get_index(None, cond)
               
                sc.pp.normalize_per_cell(adata, counts_per_cell_after=3591)
                sc.pp.log1p(adata)
                
                run_dewakss(adata, normalize=False)
                
                self.data.layers['denoised'][_idx, :] = adata.X
            
            print(f"Writing {ELIFE_SINGLE_CELL_FILE_PROCESSED}")
            self.data.write(ELIFE_SINGLE_CELL_FILE_PROCESSED)
            
    def calculate_velocities(self, force=False):
        
        if 'velocity' not in self.data.layers or force:

            self.data.layers['velocity'] = np.full_like(self.data.X, np.nan)

            for cond in self.data.obs['Condition'].dtype.categories.values:

                print(f"Calculating velocities for {cond}")

                adata = self.get_data(genotype=None, condition=cond)
                _idx = self._get_index(None, cond)
                
                ifv.global_graph(adata, verbose=True)
                
                self.data.layers['velocity'][_idx, :] = ifv.calc_velocity(
                    adata.layers['denoised'], 
                    adata.obs['cell_cycle_time'].values, 
                    adata.obsp['noise2self_distance_graph']
                )
            
            print(f"Writing {ELIFE_SINGLE_CELL_FILE_PROCESSED}")
            self.data.write(ELIFE_SINGLE_CELL_FILE_PROCESSED)

    def get_data(
        self,
        genotype="WT",
        condition="YPD"
    ):
        
        return self.data[self._get_index(genotype, condition), :].copy()
    
    def get_pseudobulk(
        self,
        genotype="WT",
        condition="YPD"
    ):
        
        if self.pseudobulk is None:
            
            self.pseudobulk = pd.DataFrame(
                self.data.layers['counts'].A,
                index=self.data.obs_names,
                columns=self.data.var_names
            )
            
            self.pseudobulk[['Genotype_Individual', 'Condition']] = self.data.obs[['Genotype_Individual', 'Condition']]
            self.pseudobulk = self.pseudobulk.groupby(
                ['Genotype_Individual', 'Condition']
            ).agg('sum')
            
            meta_df = self.pseudobulk.index.to_frame().reset_index(drop=True)
            self.pseudobulk = self.pseudobulk.reset_index(drop=True)
            
            self.pseudobulk = ad.AnnData(self.pseudobulk, dtype=np.int32)
            
            self.pseudobulk.obs = meta_df
            self.pseudobulk.obs[['Genotype_Group', 'Replicate']] = meta_df['Genotype_Individual'].str.split("_", expand=True)
            self.pseudobulk.obs['n_counts'] = self.pseudobulk.X.sum(axis=1).astype(int)
            
        return self.pseudobulk[self._get_index(genotype, condition, adata=self.pseudobulk), :].copy()

    def _get_index(
        self,
        genotype,
        condition,
        adata=None
    ):
        
        if adata is None:
            adata = self.data
        
        # Get genotype index
        if isinstance(genotype, (list, tuple)):

            _idx = adata.obs['Genotype_Group'].isin([
                "WT(ho)" if g == "WT" else g
                for g in genotype
            ])

        elif genotype is not None:
            genotype = "WT(ho)" if genotype == "WT" else genotype
            _idx = adata.obs['Genotype_Group'] == genotype

        else:
            _idx = pd.Series(True, index=adata.obs_names)
        
        # Get genotype index
        if isinstance(condition, (list, tuple)):
            _idx &= adata.obs['Condition'].isin(condition)
            
        elif condition is not None:
            _idx &= adata.obs['Condition'] == condition
            
        return _idx
    