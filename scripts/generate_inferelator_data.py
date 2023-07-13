from jtb_2022_code import FigureSingleCellData
from jtb_2022_code.figure_constants import INFERELATOR_DATA_FILE, CC_LENGTH

import gc
import anndata as ad
import numpy as np
import inferelator_velocity as ifv
import scanpy as sc

data = FigureSingleCellData()

print("Creating new data object from counts")

if 'program' in data.all_data.var:
    data.all_data.var['programs'] = data.all_data.var['program']
elif 'programs' in data.all_data.var:
    pass
else:
    data.process_programs()
    
data.all_data.obs[['UMAP_1', 'UMAP_2']] = data.all_data.obsm['X_umap'][:, 0:2]
data.all_data.obs[['PCA_1', 'PCA_2']] = data.all_data.obsm['X_pca'][:, 0:2]

_orfs = data.all_data.var_names.str.startswith(("Y", "Q", "K", "N"))

inf_data = ad.AnnData(
    data.all_data.layers['counts'].copy(),
    obs = data.all_data.obs,
    var = data.all_data.var[['CommonName', 'category', 'programs', 'leiden']],
    dtype = data.all_data.layers['counts'].dtype
)

print("Copying programs and times")
inf_data.uns['programs'] = data.all_data.uns['programs'].copy()

# Copy cell cycle time to main object
inf_data.obs['program_0_time'] = np.nan
inf_data.obs['program_1_time'] = np.nan

for _, e in data.expt_data.items():
    inf_data.obs.loc[e.obs_names, 'program_0_time'] = e.obs['program_0_time']
    inf_data.obs.loc[e.obs_names, 'program_1_time'] = e.obs['program_1_time']

# Wrap cell cycle times
_cc_time = f"program_{inf_data.uns['programs']['cell_cycle_program']}_time"

inf_data.obs.loc[inf_data.obs[_cc_time] < 0, _cc_time] = inf_data.obs.loc[inf_data.obs[_cc_time] < 0, _cc_time] + CC_LENGTH
inf_data.obs.loc[inf_data.obs[_cc_time] > CC_LENGTH, _cc_time] = inf_data.obs.loc[inf_data.obs[_cc_time]  > CC_LENGTH, _cc_time] - CC_LENGTH

# Free memory used by all that count data and whatnot
data._unload()

print("Calculating gene-program assignments")
# Get gene-program assignments based on mutual information
inf_data.var['programs'] = ifv.assign_genes_to_programs(
    inf_data,
    default_program = inf_data.uns['programs']['rapa_program'],
    default_threshold = 0.01,
    use_sparse = False,
    n_bins = 20,
    verbose = True
)

print("Creating decay & velocity layers")

inf_data.X = inf_data.X.astype(np.float32)

sc.pp.normalize_per_cell(
    inf_data,
    min_counts=0
)

# Copy decay constants and velocity from the calculated data objects
inf_data.layers['decay_constants'] = np.full(inf_data.X.shape, np.nan, dtype=np.float32)
inf_data.layers['velocity'] = np.full(inf_data.X.shape, np.nan, dtype=np.float32)
inf_data.layers['denoised'] = np.full(inf_data.X.shape, np.nan, dtype=np.float32)

PROG_KEYS = {
    inf_data.uns['programs']['rapa_program']: "rapamycin",
    inf_data.uns['programs']['cell_cycle_program']: "cell_cycle"
}

for k in data.expts:
        
    dd = data.decay_data(*k)
    _expt_idx = inf_data.obs_names.isin(dd.obs_names)

    print(f"Processing experiment {k} ({np.sum(_expt_idx)} observations)")

    _velo_rows = inf_data.layers['velocity'][_expt_idx, :]
    _decay_rows = inf_data.layers['decay_constants'][_expt_idx, :]
    _expt_metadata = inf_data.obs.loc[_expt_idx, :]
    
    inf_data.layers['denoised'][_expt_idx, :] = data.denoised_data(*k).X

    for p in PROG_KEYS.keys():

        _is_rapa = PROG_KEYS[p] == 'rapamycin'
        _p_idx = inf_data.var['programs'] == p

        print(f"Extracting values for program {PROG_KEYS[p]} ({np.sum(_p_idx)} genes)")

        # Velocity
        _velo_rows[:, _p_idx] = dd.layers[f"{PROG_KEYS[p]}_velocity"][:, _p_idx]
        
        # Decay constants based on windows
        _last_time = len(dd.uns[f"{PROG_KEYS[p]}_window_decay"]['times']) - 1
        for i, t in enumerate(dd.uns[f"{PROG_KEYS[p]}_window_decay"]['times']):
            _t_idx = np.ones(_decay_rows.shape[0], dtype=bool)
            
            # Put the left edge up if this isn't the leftmost time
            if i != 0:
                _t_idx &= _expt_metadata[f'program_{p}_time'] >= (t - 0.5)
            
            # Put the right edge up if this isn't the rightmost time
            if i != _last_time:
                _t_idx &= _expt_metadata[f'program_{p}_time'] < (t + 0.5)
                
            print(
                f"\t{i} Time {t}: {np.sum(_t_idx)} observations"
            )
            
            _t_decay = _decay_rows[_t_idx, :]
            _t_decay[:, _p_idx] = dd.varm[f"{PROG_KEYS[p]}_window_decay"][_p_idx, i].flatten()[None, :]
            _decay_rows[_t_idx, :] = _t_decay
            
            del _t_decay

    inf_data.layers['velocity'][_expt_idx, :] = _velo_rows
    inf_data.layers['decay_constants'][_expt_idx, :] = _decay_rows
    
    del _velo_rows
    del _decay_rows
    del dd
    
    print(f"Experiment extraction complete [GC: {gc.collect()}]")
    
_wt_idx = inf_data.obs['Gene'] == "WT"

print(f"{_wt_idx.sum()} observations kept (WT) from {inf_data.X.shape} data")
    
inf_data = inf_data[_wt_idx, :].copy()
   
print(f"Denoised NaN: {np.sum(np.isnan(inf_data.layers['denoised']))}")
print(f"Velocity NaN: {np.sum(np.isnan(inf_data.layers['velocity']))}")
print(f"Decay NaN: {np.sum(np.isnan(inf_data.layers['decay_constants']))}")
    
inf_data.write(INFERELATOR_DATA_FILE)
