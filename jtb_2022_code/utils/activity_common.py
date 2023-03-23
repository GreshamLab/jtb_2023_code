from inferelator.tfa.pinv_tfa import (
    ActivityOnlyPinvTFA
)
from inferelator.tfa.ridge_tfa import (
    RidgeTFA
)

from inferelator.preprocessing.velocity import (
    extract_transcriptional_output
)

from inferelator.preprocessing import PreprocessData

from inferelator.utils import InferelatorData, inferelator_verbose_level
from inferelator_velocity.times import _wrap_time

from jtb_2022_code.figure_constants import TFA_FILE, INFERELATOR_PRIORS_FILE, CC_LENGTH

import os
import gc
import anndata as ad
import numpy as np
import pandas as pd

inferelator_verbose_level(1)

PreprocessData.set_preprocessing_method(
    method_tfa='robustscaler',
    method_predictors='zscore',
    method_response='zscore',
    scale_limit_predictors=10,
    scale_limit_response=10,
    scale_limit_tfa=20
)

def calc_activity_expression(
    adata,
    prior,
    layer="X",
    velocity_layer=None,
    global_decay_constant=None,
    decay_constants=None
):
    
    prior = prior.reindex(adata.var_names, axis=0).fillna(0).astype(int)

    if velocity_layer is None:
        # Make InferelatorData object from expression and use it to get TFA
        return ActivityOnlyPinvTFA().compute_transcription_factor_activity(
            prior,
            InferelatorData(
                expression_data=adata.layers[layer] if layer != "X" else adata.X,
                gene_names=adata.var_names,
                sample_names=adata.obs_names
            ),
            keep_self=True
        ).values
    
    else:
        
        #Modifiy data to estimate transcriptional output 
        mod_data = extract_transcriptional_output(
            InferelatorData(
                expression_data=adata.layers[layer] if layer != "X" else adata.X,
                gene_names=adata.var_names,
                sample_names=adata.obs_names
            ),
            InferelatorData(
                expression_data=adata.layers[velocity_layer],
                gene_names=adata.var_names,
                sample_names=adata.obs_names
            ),
            global_decay=global_decay_constant,
            gene_and_sample_decay=decay_constants,
            decay_constant_maximum=np.log(2)
        )
        
        print("Calculating TFA")
        # Make InferelatorData object and use it to get TFA
        return ActivityOnlyPinvTFA().compute_transcription_factor_activity(
            prior,
            mod_data,
            keep_self=True
        ).values


def get_tfa(adata, layer='X', out_obsm='tfa', out_uns='tfa', recalculate=False):
    
    if out_uns in adata.uns and not recalculate:
        return adata
    
    prior = pd.read_csv(INFERELATOR_PRIORS_FILE, sep="\t", index_col=0)
    prior = prior.reindex(adata.var_names, axis=0).fillna(0).astype(int)
    prior = prior.loc[:, (prior != 0).sum(axis=0) > 0].copy()
        
    lref = adata.X if layer == 'X' else adata.layers[layer]
        
    adata.obsm[out_obsm] = ActivityOnlyPinvTFA().compute_transcription_factor_activity(
            prior,
            InferelatorData(
                expression_data=lref,
                gene_names=adata.var_names,
                sample_names=adata.obs_names
            ),
            keep_self=True
        ).values
    
    adata.uns[out_uns] = {
        'columns': prior.columns,
        'layer': layer,
        'obms_key': out_obsm
    }
    
    return adata
    

def calculate_alpha(
    adata,
    velocity_layer=None,
    decay_layer=None,
    expression_layer='X'
):
    
    expression = InferelatorData(
        expression_data=adata.layers[expression_layer] if expression_layer != "X" else adata.X,
        gene_names=adata.var_names,
        sample_names=adata.obs_names
    )
    
    if velocity_layer is not None:
        velocity_layer = InferelatorData(
            expression_data=adata.layers[velocity_layer],
            gene_names=adata.var_names,
            sample_names=adata.obs_names
        )
        
    if decay_layer is not None:
        decay_layer = InferelatorData(
            expression_data=adata.layers[decay_layer],
            gene_names=adata.var_names,
            sample_names=adata.obs_names
        )        
    
    return extract_transcriptional_output(
        expression,
        velocity_layer,
        gene_and_sample_decay=decay_layer,
        decay_constant_maximum=np.log(2)
    ).values
    
    
def get_alpha_per_cell(data_obj):
    
    alphas = ad.AnnData(
        np.full(data_obj.all_data.X.shape, np.nan, dtype=float),
        dtype=float
    )
    
    alphas.obs = data_obj.all_data.obs.copy()
    alphas.var = data_obj.all_data.var.copy()
    
    decays = data_obj.all_data.layers['decay_constants'].copy()
    
    PROG_KEYS = {
        data_obj.all_data.uns['programs']['rapa_program']: "rapamycin",
        data_obj.all_data.uns['programs']['cell_cycle_program']: "cell_cycle"
    }
    
    data_obj._unload()
    
    for e, g in data_obj.expts:
        
        _ridx = alphas.obs["Experiment"] == e
        _ridx &= alphas.obs["Gene"] == g
        _ridx = _ridx.values
        
        d = data_obj.velocity_data(e, g)
   
        for p in PROG_KEYS.keys():

                _is_rapa = PROG_KEYS[p] == 'rapamycin'
                _p_idx = alphas.var['programs'] == p
                
                alpha_estimate = extract_transcriptional_output(
                    InferelatorData(
                        expression_data=d.X[:, _p_idx],
                        gene_names=d.var_names[_p_idx],
                        sample_names=d.obs_names
                    ),
                    InferelatorData(
                        expression_data=d.layers[f"{PROG_KEYS[p]}_velocity"][:, _p_idx],
                        gene_names=d.var_names[_p_idx],
                        sample_names=d.obs_names
                    ),
                    gene_and_sample_decay=decays[_ridx, :][:, _p_idx],
                    decay_constant_maximum=np.log(2)
                ).values
                
                alpha_exists = alphas.X[_ridx, :]
                alpha_exists[:, _p_idx] = alpha_estimate
                alphas.X[_ridx, :] = alpha_exists
                
                del alpha_estimate
                del alpha_exists
                
                gc.collect()

    _wt_idx = alphas.obs['Gene'] == "WT"

    alphas = alphas[_wt_idx, :].copy()

    return alphas
                
            
def get_decay_per_cell(data_obj, by_experiment=True):
    
    print("Creating decay layer")
    # Copy decay constants and velocity from the calculated data objects
    adata = ad.AnnData(
        np.full(data_obj.all_data.X.shape, np.nan, dtype=float),
        dtype=float
    )
    
    adata.var_names = data_obj.all_data.var_names
    adata.obs_names = data_obj.all_data.obs_names
    adata.var = data_obj.all_data.var.copy()
        
    if by_experiment:
        _iter_through = [data_obj.decay_data(*k) for k in data_obj.expts]
    else:
        _iter_through = [data_obj.all_data]
        
    for dd in _iter_through:

        _expt_idx = adata.obs_names.isin(dd.obs_names)
        _expt_metadata = data_obj.all_data.obs.loc[_expt_idx, :]

        print(f"Processing experiment ({np.sum(_expt_idx)} observations)")

        adata.X[_expt_idx, :] = decay_window_to_cell_layer(dd)

    print(f"Experiment extraction complete [GC: {gc.collect()}]")

    return adata


def decay_window_to_cell_layer(
    adata,
    programs=None,
    program_keys=None
):
    
    if programs is None:
        programs = adata.var['programs']
    
    if program_keys is None:
        program_keys = {
            adata.uns['programs']['rapa_program']: "rapamycin",
            adata.uns['programs']['cell_cycle_program']: "cell_cycle"
        }
    
    _decay_rows = np.zeros(adata.shape, dtype=np.float32)
    
    for p in program_keys.keys():

            _is_rapa = program_keys[p] == 'rapamycin'
            _p_idx = programs == p           

            print(f"Extracting values for program {program_keys[p]} ({np.sum(_p_idx)} genes)")

            time_windows = [
                list(x)
                for x in list(zip(
                    adata.uns[f"{program_keys[p]}_window_decay"]['times'] - 0.5,
                    adata.uns[f"{program_keys[p]}_window_decay"]['times'] + 0.5
                ))
            ]

            if _is_rapa:
                time_windows[0][0] = None
                time_windows[-1][1] = None
            else:
                time_windows[0][0] = 0
                time_windows[-1][1] = CC_LENGTH

            _decay_rows[:, _p_idx] = _cell_decay_constants(
                adata.varm[f"{program_keys[p]}_window_decay"][_p_idx, :],
                adata.obs[f'program_{p}_time'].values,
                time_windows,
                wrap_time=None if _is_rapa else CC_LENGTH
            )

    return _decay_rows


def _cell_decay_constants(
    decays,
    times,
    time_windows,
    wrap_time=None,
    
):
    
    _decay_array = np.full(
        (times.shape[0], decays.shape[0]),
        np.nan,
        dtype=float
    )
    
    if wrap_time is not None:
        times = _wrap_time(times, wrap_time)
    
    for i, (_left, _right) in enumerate(time_windows):
        
        _t_idx = np.ones(_decay_array.shape[0], dtype=bool)
        
        if _left is not None:
            _t_idx &= times >= _left
            
        if _right is not None:
            _t_idx &= times < _right
        
        # Put the appropriate decay constants into the array
        _decay_array[_t_idx, :] = decays[:, i].ravel()[None, :]
        
    return _decay_array
