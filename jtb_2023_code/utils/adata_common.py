import anndata as _ad
import numpy as _np
import pandas as _pd
import pandas.api.types as _pat


def get_clean_anndata(
    full_adata,
    bool_idx=None,
    layer="X",
    include_pca=False,
    include_umap=False,
    replace_neighbors_with_dewakss=False,
):
    dref = full_adata.X if layer == "X" else full_adata.layers[layer]

    if bool_idx is not None:
        new_adata = _ad.AnnData(dref[bool_idx, :].copy(), dtype=dref.dtype)
        new_adata.obs = full_adata.obs.loc[bool_idx, :].copy()
    else:
        new_adata = _ad.AnnData(dref.copy(), dtype=dref.dtype)
        new_adata.obs = full_adata.obs.copy()

    new_adata.var = full_adata.var.copy()

    pca_key = layer + "_pca"
    if include_pca and pca_key in full_adata.obsm:
        pref = full_adata.obsm[pca_key]
        new_adata.obsm["X_pca"] = (
            pref[bool_idx, :].copy() if bool_idx is not None else pref.copy()
        )
        new_adata.uns["pca"] = full_adata.uns["pca"].copy()

    if replace_neighbors_with_dewakss:
        new_adata.obsp["connectivities"] = full_adata.obsp[
            "denoised_connectivities"
        ].copy()
        new_adata.obsp["distances"] = full_adata.obsp["distances"].copy()
        new_adata.uns["neighbors"] = full_adata.uns["denoised"].copy()

    return new_adata


def transfer_obs(data, key, index, values):
    if key not in data.obs:
        data.obs[key] = (
            _np.zeros(data.shape[0], dtype=values.dtype)
            if _pat.is_numeric(values)
            else ""
        )

    data.obs.loc[index, key] = values


def transfer_obsm(data, key, index, values, columns=None):
    if key not in data.obsm and columns is None:
        data.obsm[key] = _np.zeros(
            (data.shape[0], values.shape[1]),
            dtype=values.dtype
        )

    elif key not in data.obsm and columns is not None:
        data.obsm[key] = _pd.DataFrame(
            _np.zeros((data.shape[0], values.shape[1]), dtype=values.dtype),
            columns=columns,
        )

    if columns is None:
        data.obsm[key][index, :] = values
    else:
        data.obsm[key].loc[index, columns] = values


def transfer_layers(data, key, index, values):
    if key not in data.layers:
        data.layers[key] = _np.zeros((data.shape), dtype=values.dtype)

    data.layers[key][index, :] = values


def calc_times(val, t=80, reverse=False):
    if reverse:
        return _np.abs(val - 1) * t
    else:
        return val * t


def interval_normalize(arr, reverse=False):
    mi, ma = _np.min(arr), _np.max(arr)

    if mi == ma:
        return _np.zeros_like(arr)

    intervaled = (arr - mi) / (ma - mi)
    return _np.abs(1 - intervaled) if reverse else intervaled
