from inferelator_prior.velocity import decay
import numpy as np

def calc_decays(adata, include_alpha=False, add_pseudocount=True, log_expression=False, decay_quantiles=(0.01, 0.05),
                decay_key='decay', alpha_key='alpha', force=False):
    
    if decay_key in adata.var and not force:
        return adata
    
    decays, decays_se, alphas = decay.calc_decay(np.expm1(adata.layers['denoised']), 
                                                 adata.layers['denoised_velocity'], 
                                                 include_alpha=include_alpha, 
                                                 decay_quantiles=decay_quantiles, 
                                                 add_pseudocount=add_pseudocount,
                                                 log_expression=log_expression)
    
    adata.var[decay_key] = decays
    adata.var[decay_key + "_se"] = decays_se
    
    if include_alpha:
        adata.var[alpha_key] = alphas
    
    return adata
    
def calc_halflives(adata, decay_key='decay', halflife_key='halflife'):
    adata.var[halflife_key] = _halflife(adata.var[decay_key])
    
    return adata
    
def calc_decay_windows(adata, include_alpha=False, add_pseudocount=False, log_expression=False, decay_quantiles=(0.00, 0.05), 
                       bootstrap=True, force=False):
    
    if 'decay_windows' in adata.varm and not force:
        return adata
    
    decays, decays_se, a, t_c = decay.calc_decay_sliding_windows(np.expm1(adata.layers['denoised']), 
                                                                 adata.layers['denoised_velocity'], 
                                                                 adata.obs['time_pca_pt'].values,
                                                                 include_alpha=include_alpha, 
                                                                 decay_quantiles=decay_quantiles, 
                                                                 add_pseudocount=add_pseudocount, 
                                                                 log_expression=log_expression, 
                                                                 centers=np.linspace(0.5, 79.5, 80),
                                                                 width=1.,
                                                                 bootstrap_estimates=bootstrap)
    
    adata.uns['decay_windows'] = {'params': {'include_alpha': include_alpha, 
                                             'add_pseudocount': add_pseudocount, 
                                             'log_expression': log_expression, 
                                             'decay_quantiles': list(decay_quantiles),
                                             'bootstrap': bootstrap},
                                  'times': t_c}

    adata.varm['decay_windows'] = np.array(decays).T
    adata.varm['decay_windows_se'] = np.array(decays_se).T
    
    if include_alpha:
        adata.varm['output_alpha'] = np.array(a).T
        
    return adata
    
def _halflife(decay_constants):
    
    hl = np.full_like(decay_constants, np.nan)
    np.divide(np.log(2), decay_constants, where=decay_constants != 0, out=hl)
    hl[np.isinf(hl)] = np.nan
    return hl