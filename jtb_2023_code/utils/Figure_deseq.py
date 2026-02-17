import pydeseq2
import pydeseq2.dds
import pydeseq2.ds

import scipy.sparse as sps
import numpy as np
import pandas as pd


def run_deseq(
    data,
    obs_col,
    base_value,
    lfc_null=np.log2(1.2),
    layer='X',
    size_factors='ratio',
    quiet=True
):

    lref = data.X if layer == 'X' else data.layers[layer]

    deseq_data = pydeseq2.dds.DeseqDataSet(
        counts=lref.toarray() if sps.issparse(lref) else lref,
        metadata=data.obs[[obs_col]],
        design_factors=[obs_col],
        refit_cooks=True,
        quiet=quiet
    )

    deseq_data.fit_size_factors(size_factors)
    deseq_data.fit_genewise_dispersions()
    deseq_data.fit_dispersion_trend()
    deseq_data.fit_dispersion_prior()
    deseq_data.fit_MAP_dispersions()
    deseq_data.fit_LFC()
    deseq_data.calculate_cooks()
    deseq_data.refit()

    def _get_results(comp):
        result = pydeseq2.ds.DeseqStats(
            deseq_data,
            alpha=0.05,
            cooks_filter=True,
            independent_filter=True,
            contrast=[obs_col.replace("_", "-"), comp, base_value],
            lfc_null=lfc_null,
            alt_hypothesis='greaterAbs'
        )
        result.run_wald_test()
        result._p_value_adjustment()
        result.summary()
    
        result = result.results_df[['baseMean', 'log2FoldChange', 'pvalue', 'padj']].copy()
        result[obs_col] = comp
        result['Gene'] = data.var_names
        return result

    return pd.concat(
        [
            _get_results(x)
            for x in data.obs[obs_col].cat.categories
            if x != base_value
        ]
    )
