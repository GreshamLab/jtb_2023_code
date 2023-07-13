from jtb_2022_code.figure_constants import (
    INFERELATOR_DATA_FILE,
    CC_LENGTH,
    LATENT_DATA_FILES,
    AGGREGATE_DATA_FILES
)

from inferelator_velocity.utils import aggregate_sliding_window_times
import anndata as ad
import pandas as pd
import numpy as np
import gc

print(f"Loading {INFERELATOR_DATA_FILE}")
inf_data = ad.read(INFERELATOR_DATA_FILE)

PROG_KEYS = {
    inf_data.uns['programs']['rapa_program']: "rapamycin",
    inf_data.uns['programs']['cell_cycle_program']: "cell_cycle"
}

inf_obs = inf_data.obs.copy()
inf_var = inf_data.var.copy()

print(f"Loading latent files")
data_files = {
    k: pd.read_csv(v, sep="\t", index_col=0)
    for k, v in LATENT_DATA_FILES.items()
}

centers = {
    "rapamycin": np.arange(-10, 60) + 0.5,
    "cell_cycle": np.arange(0, CC_LENGTH) + 0.5
}

def _aggregate_genes(df, genes, times, prog_name, width=1, axis=0):
    
    _genes = genes.intersection(df.columns)
    _data = df[_genes].values
    
    aggs, _centers = aggregate_sliding_window_times(
        _data,
        times,
        centers=centers[prog_name],
        width=width,
        agg_func=np.median,
        agg_kwargs={'axis': axis}
    )
    
    median = pd.DataFrame(aggs.T, index=_genes, columns=_centers)
    median['agg_func'] = 'median'
    
    aggs, _centers = aggregate_sliding_window_times(
        _data,
        times,
        centers=centers[prog_name],
        width=width,
        agg_func=np.std,
        agg_kwargs={'axis': axis}
    )
    
    stdev = pd.DataFrame(aggs.T, index=_genes, columns=_centers)
    stdev['agg_func'] = 'stdev'
    
    return pd.concat((median, stdev))

for p, prog_name in PROG_KEYS.items():
    
    print(f"Processing program {prog_name}")
    
    _genes = inf_var.index[inf_var['programs'] == p]
    _times = inf_obs[f'program_{p}_time']
    
    results = []
    
    for expt in [1, 2, None]:
        
        if expt is not None:
            _expt_idx = inf_obs['Experiment'] == expt
        else:
            _expt_idx = pd.Series(True, index=inf_obs.index)

        # Get standard layers
        for layer in ["expression", "velocity", "denoised"]:

            print(f"Aggregating {layer}")

            agg_result = _aggregate_genes(
                inf_data[_expt_idx, inf_var['programs'] == p].to_df(
                    layer=layer if layer != "expression" else None
                ),
                _genes,
                _times[_expt_idx],
                prog_name
            )

            agg_result.index.name = "gene"
            agg_result["dataset"] = layer
            agg_result['experiment'] = expt if expt is not None else "ALL"

            print(f"Aggregation complete: {agg_result.shape}")
            print(agg_result.iloc[0, :])

            results.append(agg_result)
            
        # Get decay constants from already aggregated data
        # Stored in VARM
        if expt is None:
            _varm_key = f"{prog_name}_window_decay"
        else:
            _varm_key = f"{prog_name}_window_decay_{expt}"

        agg_result = pd.DataFrame(
            inf_data.varm[_varm_key][inf_var['programs'] == p, :],
            index=_genes,
            columns=centers[prog_name]
        )
        
        agg_result['dataset'] = 'decay_constants'
        agg_result['experiment'] = expt if expt is not None else "ALL"
        
        results.append(agg_result)
            
        # Get latent layers
        for k, v in data_files.items():

            print(f"Aggregating {prog_name} {k}:")

            agg_result = _aggregate_genes(
                v.loc[_expt_idx.values, :],
                _genes,
                _times[_expt_idx],
                prog_name
            )

            agg_result.index.name = "gene"
            agg_result["dataset"] = k
            agg_result['experiment'] = expt if expt is not None else "ALL"

            print(f"Aggregation complete: {agg_result.shape}")
            print(agg_result.iloc[0, :])

            results.append(agg_result)

    results = pd.concat(results)

    print(f"Writing {results.shape} to {AGGREGATE_DATA_FILES[prog_name]}")

    results.to_csv(
        AGGREGATE_DATA_FILES[prog_name],
        sep="\t"
    )
        
    
    
