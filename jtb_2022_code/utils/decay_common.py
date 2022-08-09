from inferelator_velocity import decay
import numpy as np

def calc_decays(adata, velocity_key, output_key='decay_windows', output_alpha_key='output_alpha',
                include_alpha=False, decay_quantiles=(0.00, 0.05),
                layer="X", force=False):
    
    if output_key in adata.var and not force:
        return adata
    
    lref = adata.X if layer == "X" else adata.layers[layer]

    decays, decays_se, alphas = _calc_decay(lref, 
                                            adata.layers[velocity_key], 
                                            include_alpha=include_alpha, 
                                            decay_quantiles=decay_quantiles)
    
    adata.var[output_key] = decays
    adata.var[output_key + "_se"] = decays_se
    
    if include_alpha:
        adata.var[alpha_key] = alphas
    
    return adata
    
def calc_halflives(adata, decay_key='decay', halflife_key='halflife'):
    adata.var[halflife_key] = _halflife(adata.var[decay_key])
    
    return adata
    
def calc_decay_windows(adata, velocity_key, time_key, output_key='decay_windows', output_alpha_key='output_alpha',
                       include_alpha=False, decay_quantiles=(0.00, 0.05), 
                       bootstrap=False, force=False, layer='X', t_min=0, t_max=80):
    
    if output_key in adata.varm and not force:
        return adata
    
    lref = adata.X if layer == "X" else adata.layers[layer]
    
    decays, decays_se, a, t_c = _calc_decay_windowed(lref, 
                                                     adata.layers[velocity_key], 
                                                     adata.obs[time_key].values,
                                                     include_alpha=include_alpha, 
                                                     decay_quantiles=decay_quantiles,
                                                     t_min=t_min,
                                                     t_max=t_max,
                                                     bootstrap=bootstrap)
    
    adata.uns[output_key] = {'params': {'include_alpha': include_alpha, 
                                        'decay_quantiles': list(decay_quantiles),
                                        'bootstrap': bootstrap},
                             'times': t_c}

    adata.varm[output_key] = np.array(decays).T
    adata.varm[output_key + "_se"] = np.array(decays_se).T
    
    if include_alpha:
        adata.varm[output_alpha_key] = np.array(a).T
        
    return adata

def _calc_decay(expr, velo, include_alpha=False, decay_quantiles=(0.0, 0.05)):
    
    return decay.calc_decay(expr, velo, include_alpha=include_alpha, decay_quantiles=decay_quantiles)

def _calc_decay_windowed(expr, velo, times, include_alpha=False, decay_quantiles=(0.00, 0.05), 
                         bootstrap=True, t_min=0, t_max=80, time_wrap=None):
    
    if time_wrap is not None:
        times = times.copy()
        times[times < 0] = times[times < 0] + time_wrap
        times[times > time_wrap] = times[times > time_wrap] - time_wrap

    
    return decay.calc_decay_sliding_windows(expr, 
                                            velo, 
                                            times,
                                            include_alpha=include_alpha, 
                                            decay_quantiles=decay_quantiles,
                                            centers=np.linspace(t_min + 0.5, t_max - 0.5, int(t_max - t_min)),
                                            width=1.,
                                            bootstrap_estimates=bootstrap)    
    
def _halflife(decay_constants):
    
    hl = np.full_like(decay_constants, np.nan)
    np.divide(np.log(2), decay_constants, where=decay_constants != 0, out=hl)
    hl[np.isinf(hl)] = np.nan
    return hl