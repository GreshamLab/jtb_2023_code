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

from scself import standardize_data

from ..figure_constants import (
    RAPA_SINGLE_CELL_EXPR_BY_EXPT,
    RAPA_SINGLE_CELL_EXPR_PROCESSED,
    RAPA_SINGLE_CELL_EXPR_FILE,
    RAPA_SINGLE_CELL_VELOCITY_BY_EXPT,
    RAPA_SINGLE_CELL_DENOISED_BY_EXPT,
    N_PCS,
    N_NEIGHBORS,
    UMAP_NPCS,
    UMAP_NNS,
    UMAP_MIN_DIST,
    VERBOSE,
    PSEUDOTIME_FILES,
    DECAY_CONSTANT_FILES,
    GENE_NAMES_FILE,
    GENE_GROUP_FILE,
    CC_TIME_COL,
    RAPA_TIME_COL,
    CC_TIME_ORDER,
    RAPA_TIME_ORDER,
    CC_LENGTH,
    CC_COLS,
    AGG_COLS,
    CELLCYCLE_GROUP_COL,
    OTHER_GROUP_COL,
    RAPA_BULK_EXPR_FILE,
    RAPA_BULK_EXPR_FILE_META_DATA_COLS,
    RAPA_BULK_EXPR_FILE_TIMES
)

from .projection_common import (
    do_pca,
    do_umap
)
from .decay_common import (
    calc_decays,
    calc_decay_windows,
    _calc_decay_windowed,
    _calc_decay,
    get_decay_per_cell,
    decay_window_to_cell_layer,
)

from .velocity_common import calculate_velocities
from .pseudotime_common import spearman_rho_pools, get_pca_pt
from .process_published_data import process_all_decay_links
from sklearn.metrics import pairwise_distances

NO_RP_DEPTH = 2000


class FigureSingleCellData:
    _all_data = None
    _expt_data = None
    _expt_keys = None

    _pseudotime_rho = None
    _max_pseudotime_rho = None

    expt_cats = [1, 2]
    gene_cats = ["WT", "fpr1"]

    @property
    def expt_files(self):
        return {
            (e, g): RAPA_SINGLE_CELL_EXPR_BY_EXPT.format(e=e, g=g)
            for e in self.expt_cats
            for g in self.gene_cats
        }

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
        return all(
            ["X_pca" in self.all_data.obsm]
            + ["X_pca" in v.obsm for k, v in self.expt_data.items()]
        )

    @property
    def has_umap(self):
        return all(
            ["X_umap" in self.all_data.obsm]
            + ["X_umap" in v.obsm for k, v in self.expt_data.items()]
        )

    @property
    def pseudotime(self):
        if "pseudotime" not in self.all_data.obsm:
            self.load_pseudotime()

        data = self.all_data.obsm["pseudotime"].copy()
        data.columns = _pd.MultiIndex.from_frame(
            self.all_data.uns["pseudotime_columns"]
        )

        return data

    @property
    def pseudotime_rho(self):
        if self._pseudotime_rho is None:
            self.calculate_pseudotime_rho()

        return self._pseudotime_rho.copy()

    @property
    def max_pseudotime_rho(self):
        if self._max_pseudotime_rho is None:
            self.calculate_max_pseudotime_rho()

        return self._max_pseudotime_rho.copy()

    @property
    def optimal_pseudotime_rho(self):
        ptr = self.pseudotime_rho.loc[:, (slice(None), False, slice(None))]
        ptr = ptr.T.loc[:, (slice(None), "WT")]
        ptr = ptr.reset_index().droplevel(1, axis=1)
        ptr["mean_rho"] = ptr[[1, 2]].mean(axis=1)
        _opt_idx = (
            ptr[["method", "mean_rho"]].groupby(
                ["method"]
            ).transform('max')["mean_rho"]
            == ptr["mean_rho"]
        )
        return ptr[_opt_idx].set_index("method", drop=True)

    @property
    def _all_adatas(self):
        return [self.all_data] + [v for k, v in self.expt_data.items()]

    def __init__(self, start_from_scratch=False, load_expts=True):
        self._load(from_unprocessed=start_from_scratch, load_expts=load_expts)

    def do_projections(self):
        if not self.has_pca:
            self.apply_inplace_to_everything(do_pca, max(N_PCS))

        if not self.has_umap:
            self.apply_inplace_to_everything(
                do_umap, UMAP_NPCS, UMAP_NNS, UMAP_MIN_DIST
            )
            self.save()

    def apply_to_expts(self, func, *args, **kwargs):
        return [
            self._apply(x, func, *args, **kwargs)
            for _, x in self.expt_data.items()
        ]

    def apply_to_everything(self, func, *args, **kwargs):
        return [
            self._apply(x, func, *args, **kwargs)
            for x in self._all_adatas
        ]

    def apply_inplace_to_expts(self, func, *args, **kwargs):
        self.apply_to_expts(func, *args, **kwargs)

    def apply_inplace_to_everything(self, func, *args, **kwargs):
        self.apply_to_everything(func, *args, **kwargs)

    @staticmethod
    def _apply(data, func, *args, **kwargs):
        if VERBOSE > 1:
            print(
                f"Applying {func.__name__} to data [ "
                f'{str(data.obs["Experiment"].unique().astype(str))}, '
                f'{str(data.obs["Gene"].unique().astype(str))}]'
                f" {data.shape}"
            )
        return func(data, *args, **kwargs)

    def _load(self, from_unprocessed=False, load_expts=True):
        _first_load = from_unprocessed or not _os.path.exists(
            RAPA_SINGLE_CELL_EXPR_PROCESSED
        )
        fn = (
            RAPA_SINGLE_CELL_EXPR_PROCESSED
            if not _first_load
            else RAPA_SINGLE_CELL_EXPR_FILE
        )

        if VERBOSE:
            print(f"Reading Single Cell Data from {fn}")

        self._all_data = _ad.read_h5ad(fn)

        if VERBOSE and _first_load:
            print(
                "Initial loading and preprocessing of raw data "
                f"{self._all_data.shape}"
            )

        if _first_load:
            self._all_data = self._first_load(self._all_data)

        if load_expts:
            self._load_expts(force_extraction_from_all=from_unprocessed)

        if _first_load:
            self.apply_inplace_to_everything(
                self._normalize,
                n_counts=NO_RP_DEPTH
            )
            self.save()

    def _unload(self):
        del self._all_data
        del self._expt_data

        _gc.collect()

        self._all_data = None
        self._expt_data = None

    def _load_expts(self, force_extraction_from_all=False):
        if self.expt_data is not None:
            return

        self._expt_keys = [k for k in self.expt_files]
        self._expt_data = {k: None for k in self._expt_keys}

        for k, v in self.expt_files.items():
            if _os.path.exists(v) and not force_extraction_from_all:
                if VERBOSE:
                    print(f"Reading Single Cell Experiment Data from {v}")
                self._expt_data[k] = _ad.read_h5ad(v)

            else:
                e, g = k
                if VERBOSE:
                    print(f"Extracting [{k}] Single Cell Data from all data")

                self._expt_data[k] = self.all_data[
                    (self.all_data.obs["Experiment"] == e)
                    & (self.all_data.obs["Gene"] == g),
                    :,
                ].copy()

    @staticmethod
    def _normalize(adata, n_counts=NO_RP_DEPTH, method='log'):

        if 'counts' not in adata.layers.keys():
            adata.layers["counts"] = adata.X.copy()

        standardize_data(
            adata,
            method=method,
            target_sum=n_counts,
            subset_genes_for_depth=~(
                adata.var['RP'] |
                adata.var['RiBi']
            )
        )

        if (method == 'scale') or (method == 'log_scale'):
            adata.var['scale_factor'] = adata.var['X_scale_factor']

        return adata

    def save(self):
        # Save all data
        if self.all_data is not None:
            if VERBOSE:
                print(
                    "Writing Single Cell Data to "
                    f"{RAPA_SINGLE_CELL_EXPR_PROCESSED}"
                )
            self._all_data.write(RAPA_SINGLE_CELL_EXPR_PROCESSED)

        # Save individual experiments
        if self.expt_data is not None:
            for k, v in self.expt_files.items():
                if self.expt_data[k] is not None:
                    if VERBOSE:
                        print(f"Writing Single Cell Data to {v}")
                    self.expt_data[k].write(v)

    def load_pseudotime(self, files=PSEUDOTIME_FILES, reload=False):

        if reload or 'pseudotime' not in self.all_data.obsm:
            # LOAD FLATFILES #
            print("Loading pseudotime flatfiles")
            loaded_data = [
                _pd.read_csv(fn, sep="\t", index_col=0 if has_idx else None)
                for k, (fn, has_idx) in PSEUDOTIME_FILES.items()
            ]

            for i, k in enumerate(PSEUDOTIME_FILES.keys()):
                loaded_data[i].index = loaded_data[i].index.astype(str)
                loaded_data[i].columns = _pd.MultiIndex.from_tuples(
                    [(k[0], k[1], c) for c in loaded_data[i].columns],
                    names=("method", "denoised", "values"),
                )

                if k[0] == "palantir" and not k[1]:
                    loaded_data[i] = _select_palantir_dcs(loaded_data[i])

            print("Calculating PCA pseudotimes")
            for i in range(2):
                pca_pt = _pd.DataFrame(
                    _np.nan,
                    index=self.all_data.obs_names,
                    columns=_pd.MultiIndex.from_tuples(
                        [("pca", i == 1, "pca")]
                    ),
                    dtype=float,
                )

                for k in self.expts:

                    _ref = self.expt_data[k]

                    if (
                        (i == 1) and
                        'denoised_pca' not in _ref.obsm.keys()
                    ):
                        _ref.obsm['denoised_pca'] = self.denoised_data(
                            *k
                        ).obsm['X_pca'][:, 0:5]

                    pca_pt.loc[self._all_data_expt_index(*k), :] = get_pca_pt(
                        _ref,
                        pca_key="X_pca" if i == 0 else "denoised_pca",
                    ).reshape(-1, 1)

                loaded_data.append(pca_pt)

            loaded_data = _pd.concat(loaded_data, axis=1)

            # Store column names in a dataframe
            self.all_data.uns["pseudotime_columns"] = _pd.DataFrame(
                loaded_data.columns.to_flat_index().tolist()
            )

            self.all_data.uns["pseudotime_columns"].index = self.all_data.uns[
                "pseudotime_columns"
            ].index.astype(str)
            self.all_data.uns["pseudotime_columns"].columns = (
                "method",
                "denoised",
                "values",
            )

            loaded_data.columns = _pd.Index(
                list(range(len(loaded_data.columns)))
            ).astype(
                str
            )

            self.all_data.obsm["pseudotime"] = loaded_data

            self.save()

        return self

    def calculate_pseudotime_rho(self):
        print("Calculating spearman rho for pseudotimes")

        pt = self.pseudotime
        rho = []

        for k in self.expts:
            _idx = self._all_data_expt_index(*k)
            df = _pd.DataFrame(
                pt.loc[_idx, :].apply(
                    lambda x: spearman_rho_pools(
                        self.all_data.obs.loc[_idx, "Pool"], x
                    ),
                    raw=True,
                )
            ).T
            df.index = _pd.MultiIndex.from_tuples(
                [k],
                names=("Experiment", "Gene")
            )

            rho.append(_np.abs(df))

        self._pseudotime_rho = _pd.concat(rho)

    def calculate_max_pseudotime_rho(self):
        max_rho = self.pseudotime_rho.groupby(
            axis=1,
            level=["method", "denoised"]
        ).agg(
            _np.max
        )

        max_rho[("time", False)] = 0.0
        max_rho[("time", True)] = 0.0

        for k in self.expts:
            expt_ref = self.expt_data[k]
            max_rho.loc[k, ("time", False)] = _np.abs(
                spearman_rho_pools(
                    expt_ref.obs["Pool"],
                    expt_ref.obs["program_rapa_time"]
                )
            )
            max_rho.loc[k, ("time", True)] = _np.abs(
                spearman_rho_pools(
                    expt_ref.obs["Pool"],
                    expt_ref.obs["program_rapa_time_denoised"]
                )
            )

        self._max_pseudotime_rho = max_rho

    def load_published_decay(self):
        if not all(
            x in self.all_data.var.columns
            for x in DECAY_CONSTANT_FILES.keys()
        ):
            p_decay = process_all_decay_links(self.all_data.var_names)
            self.all_data.var[p_decay.columns] = p_decay

    def velocity_data(self, expt, gene):
        _fn = RAPA_SINGLE_CELL_VELOCITY_BY_EXPT.format(e=expt, g=gene)

        if _os.path.exists(_fn):
            print(f"Loading velocity data from {_fn}")
            return _ad.read_h5ad(_fn)

        else:
            print(f"{_fn} not found. Generating velocity data:")

            _eref = self.expt_data[(expt, gene)]

            # Reset the obs data
            v_adata = _ad.AnnData(
                self.denoised_data(expt, gene).X.astype(_np.float32),
                obs=_eref.obs.copy(),
                var=_eref.var.copy()
            )

            v_adata.obsp["noise2self_distance_graph"] = (
                _eref.obsp["noise2self_distance_graph"].copy()
            )

            calculate_velocities(v_adata)
            v_adata.write(_fn)

            return v_adata

    def decay_data(self, expt, gene, recalculate=False):
        _velo_data = self.velocity_data(expt, gene)

        if "programs" not in _velo_data.uns:
            self._transfer_programs(_velo_data)
            _fn = RAPA_SINGLE_CELL_VELOCITY_BY_EXPT.format(e=expt, g=gene)

            if VERBOSE:
                print(f"Writing Single Cell Decay Data to {_fn}")

            _velo_data.write(_fn)

        if "cell_cycle_decay" not in _velo_data.var.columns or recalculate:
            print("Calculating Biophysical Paramaters:")

            for g_key, time_key, tmin, tmax in [
                ("cell_cycle", CC_TIME_COL, 0, 88),
                ("rapamycin", RAPA_TIME_COL, -10, 60),
            ]:
                calc_decays(
                    _velo_data,
                    g_key + "_velocity",
                    output_key=g_key + "_decay",
                    output_alpha_key=g_key + "_alpha",
                    force=recalculate,
                )

                if g_key == "cell_cycle":
                    _vdata = _velo_data[
                        _velo_data.obs["Pool"].isin([1, 2]),
                        :
                    ].copy()
                else:
                    _vdata = _velo_data

                _out_key = g_key + "_window_decay"

                calc_decay_windows(
                    _vdata,
                    g_key + "_velocity",
                    time_key,
                    output_key=_out_key,
                    output_alpha_key=g_key + "_window_alpha",
                    include_alpha=False,
                    t_min=tmin,
                    t_max=tmax,
                    force=recalculate,
                )

                _velo_data.uns[_out_key] = _vdata.uns[_out_key]
                _velo_data.varm[_out_key] = _vdata.varm[_out_key]

            _pref = self.all_data.uns["programs"]
            _velo_data.layers["decay_constants"] = decay_window_to_cell_layer(
                _velo_data,
                programs=self.all_data.var["programs"],
                program_keys={
                    _pref["rapa_program"]: "rapamycin",
                    _pref["cell_cycle_program"]: "cell_cycle",
                },
            )

            _fn = RAPA_SINGLE_CELL_VELOCITY_BY_EXPT.format(e=expt, g=gene)

            if VERBOSE:
                print(f"Writing Single Cell Decay Data to {_fn}")

            _velo_data.write(_fn)

        return _velo_data

    def decay_data_all(self, recalculate=False, reextract=False):
        if recalculate or "cell_cycle_decay" not in self.all_data.var.columns:
            adata_list = [self.decay_data(1, "WT"), self.decay_data(2, "WT")]

            for g_key, time_key, tmin, tmax in [
                ("cell_cycle", CC_TIME_COL, 0, 88),
                ("rapamycin", RAPA_TIME_COL, -10, 60),
            ]:
                try:
                    expr = _np.vstack([a.X.A for a in adata_list])
                except AttributeError:
                    expr = _np.vstack([a.X for a in adata_list])

                velo = _np.vstack([
                    a.layers[g_key + "_velocity"]
                    for a in adata_list
                ])
                times = _np.hstack([
                    a.obs[time_key].values
                    for a in adata_list
                ])

                decays, decays_se, a = _calc_decay(
                    expr,
                    velo,
                    include_alpha=False
                )

                self.all_data.var[g_key + "_decay"] = decays
                self.all_data.var[g_key + "_decay_se"] = decays_se
                self.all_data.var[g_key + "_alpha"] = a

                self.all_data.uns[g_key + "_decay"] = {
                    "params": {
                        "include_alpha": False,
                        "decay_quantiles": [0.0, 0.05],
                        "bootstrap": False,
                    }
                }

                decays, decays_se, a, t_c = _calc_decay_windowed(
                    expr,
                    velo,
                    times,
                    t_min=tmin,
                    t_max=tmax,
                    include_alpha=False,
                    bootstrap=False,
                )

                self.all_data.uns[g_key + "_window_decay"] = {
                    "params": {
                        "include_alpha": True,
                        "decay_quantiles": [0.0, 0.05],
                        "bootstrap": False,
                    },
                    "times": t_c,
                }

                _varm = self.all_data.varm

                _varm[g_key + "_window_decay"] = _np.array(decays).T
                _varm[g_key + "_window_decay_se"] = _np.array(decays_se).T
                _varm[g_key + "_window_alpha"] = _np.array(a).T

            del adata_list
            _gc.collect()

            self.all_data.layers["decay_constants"] = get_decay_per_cell(
                self, by_experiment=False
            ).X

            if VERBOSE:
                print(
                    "Writing Single Cell Data to "
                    f"{RAPA_SINGLE_CELL_EXPR_PROCESSED}"
                )

            self._all_data.write(RAPA_SINGLE_CELL_EXPR_PROCESSED)

        elif reextract:
            for g_key, time_key, tmin, tmax in [
                ("cell_cycle", CC_TIME_COL, 0, 88),
                ("rapamycin", RAPA_TIME_COL, -10, 60),
            ]:
                self.all_data.layers["decay_constants"] = get_decay_per_cell(
                    self, by_experiment=False
                ).X

            if VERBOSE:
                print(
                    "Writing Single Cell Data to "
                    f"{RAPA_SINGLE_CELL_EXPR_PROCESSED}"
                )

            self._all_data.write(RAPA_SINGLE_CELL_EXPR_PROCESSED)

        return self.all_data

    def denoised_data(self, expt, gene):
        _fn = RAPA_SINGLE_CELL_DENOISED_BY_EXPT.format(e=expt, g=gene)

        if _os.path.exists(_fn):
            print(f"Loading denoised data from {_fn}")
            return _ad.read_h5ad(_fn)

        else:
            print(f"{_fn} not found. Generating denoised data:")
            adata = _ad.AnnData(
                self.expt_data[(expt, gene)].layers["counts"].copy(),
                obs=self.expt_data[(expt, gene)].obs.copy(),
                var=self.expt_data[(expt, gene)].var.copy()
            )

            self.process_programs()

            adata.obsp["noise2self_distance_graph"] = (
                self.expt_data[(expt, gene)].obsp[
                    "noise2self_distance_graph"
                ].copy()
            )

            adata = self._normalize(
                adata,
                method='depth',
                n_counts=NO_RP_DEPTH
            )

            _ifv.denoise(
                adata,
                output_layer='X'
            )

            _sc.pp.pca(adata, n_comps=100)

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

        _orf_idx = adata.var_names.str.startswith("Y")
        _orf_idx |= adata.var_names.str.startswith("Q")
        _orf_idx |= adata.var_names.str.endswith("MX")

        print(f"Removing {adata.shape[1] - _orf_idx.sum()} non-coding")
        adata = adata[:, _orf_idx].copy()

        # Copy counts and save basic count depth stats
        adata.layers["counts"] = adata.X.copy()

        _adata_obs_sum = adata.X.sum(axis=1).astype(int)
        _adata_obs_n = adata.X.astype(bool).sum(axis=1)
        _adata_var_sum = adata.X.sum(axis=0).astype(int)

        try:
            adata.obs["n_counts"] = _adata_obs_sum.A1
            adata.var["n_counts"] = _adata_var_sum.A1
            adata.obs["n_genes"] = _adata_obs_n.A1
        except AttributeError:
            adata.obs["n_counts"] = _adata_obs_sum
            adata.var["n_counts"] = _adata_var_sum
            adata.obs["n_genes"] = _adata_obs_n

        # Fix categorical dtypes
        adata.obs["Pool"] = adata.obs["Pool"].astype("category")
        adata.obs["Experiment"] = adata.obs["Experiment"].astype("category")
        adata.obs["Gene"] = adata.obs["Gene"].astype("category")

        adata = _gene_metadata(adata)
        adata = _call_cc(adata)
        adata = calc_group_props(adata)
        adata = calc_other_cc_groups(adata)

        _add_broad_category(adata)

        return adata

    @staticmethod
    def gene_common_name(gene_symbol):
        return CommonNames.common_name(gene_symbol)

    def process_programs(self, recalculate=False):
        if recalculate or "programs" not in self.all_data.uns:
            _ifv.program_select(
                self.all_data,
                layer="counts",
                metric="cosine",
                verbose=True,
                standardization_method='scale',
                standardization_kwargs=dict(
                    target_sum=NO_RP_DEPTH,
                    subset_genes_for_depth=~(
                        self.all_data.var['RP'] |
                        self.all_data.var['RiBi']
                    )
                ),
                filter_to_hvg=True
            )

            _rp_program = self.all_data.var.loc[
                self.all_data.var['RP'],
                "programs"
            ].value_counts().index[0]

            if _rp_program == '0':
                _rapa_program, _cc_program = "0", "1"
            else:
                _rapa_program, _cc_program = "1", "0"

            self.all_data.uns["programs"]["rapa_program"] = _rapa_program
            self.all_data.uns["programs"]["cell_cycle_program"] = _cc_program

            self.apply_inplace_to_expts(self._transfer_programs)

        self.process_graphs()
        self.process_times(recalculate=recalculate)

    def process_graphs(self, recalculate=False):
        if (
            recalculate or
            "noise2self_distance_graph" not in self.expt_data[(1, "WT")].obsp
        ):
            self.apply_inplace_to_expts(
                _ifv.global_graph,
                neighbors=N_NEIGHBORS,
                npcs=N_PCS,
                connectivity=True,
                verbose=True,
                standardization_method='scale'
            )

            self.save()
            self.process_times(recalculate=True)

    def process_times(self, recalculate=False):
        _rapa_program = self.all_data.uns["programs"]["rapa_program"]
        _cc_program = self.all_data.uns["programs"]["cell_cycle_program"]

        if recalculate or "program_0_time" not in self.all_data.obs.columns:
            self.apply_inplace_to_expts(self._transfer_programs)

            self.apply_inplace_to_expts(
                _ifv.times.program_times,
                {_rapa_program: "Pool_Combined", _cc_program: "CC"},
                {_rapa_program: RAPA_TIME_ORDER, _cc_program: CC_TIME_ORDER},
                layer="counts",
                wrap_time={_rapa_program: None, _cc_program: CC_LENGTH},
                verbose=True,
                standardization_method='scale',
                mcv_kwargs=dict(standardization_method='scale'),
                nan_on_error=True
            )

            def _sort_out_times(adata):
                adata.obs[RAPA_TIME_COL] = adata.obs[
                    f"program_{_rapa_program}_time"
                ].copy()
                adata.obs[CC_TIME_COL] = adata.obs[
                    f"program_{_cc_program}_time"
                ].copy()

            self.apply_inplace_to_expts(_sort_out_times)

            # Move WT times back to all_data object
            # Leave fdr1 as NaN
            self.all_data.obs[f"program_{_rapa_program}_time"] = _np.nan
            self.all_data.obs[f"program_{_cc_program}_time"] = _np.nan

            for k in self.expts:
                if k[1] != "WT":
                    continue

                _idx = self._all_data_expt_index(*k)

                self.all_data.obs.loc[
                    _idx, f"program_{_rapa_program}_time"
                ] = self.expt_data[k].obs[f"program_{_rapa_program}_time"]
                self.all_data.obs.loc[
                    _idx, f"program_{_cc_program}_time"
                ] = self.expt_data[k].obs[f"program_{_cc_program}_time"]

            # Assign the remaining genes
            self.all_data.var["programs"] = _ifv.assign_genes_to_programs(
                self.all_data,
                default_program=self.all_data.uns["programs"]["rapa_program"],
                default_threshold=0.1,
                use_sparse=False,
                n_bins=20,
                verbose=True,
                layer="counts",
                return_mi=False,
                standardization_method='scale'
            )

            self.save()

        _dd_time_col = f"{RAPA_TIME_COL}_denoised"

        if recalculate or _dd_time_col not in self.all_data.obs.columns:
            self.all_data.obs[f"{RAPA_TIME_COL}_denoised"] = _np.nan
            self.all_data.obs[f"{CC_TIME_COL}_denoised"] = _np.nan

            for k in self.expts:
                _all = self.all_data
                _expt = self.expt_data[k]
                _denoised = self.denoised_data(*k)

                _cc = _all.uns['programs']['cell_cycle_program']
                _rapa = _all.uns['programs']['rapa_program']

                _n_comps = {
                    _rapa: len(
                        _expt.uns[f"program_{_rapa}_pca"]["variance"]
                    ),
                    _cc: len(
                        _expt.uns[f"program_{_cc}_pca"]["variance"]
                    ),
                }

                self._transfer_programs(_denoised)
                _ifv.program_times(
                    _denoised,
                    {
                        _rapa: "Pool_Combined",
                        _cc: "CC",
                    },
                    {
                        _rapa: RAPA_TIME_ORDER,
                        _cc: CC_TIME_ORDER,
                    },
                    wrap_time={_rapa_program: None, _cc_program: CC_LENGTH},
                    layer="X",
                    n_comps=_n_comps,
                    nan_on_error=True
                )

                _expt.obs[f"{CC_TIME_COL}_denoised"] = _denoised.obs[
                    f"program_{_cc_program}_time"
                ]
                _expt.obs[f"{RAPA_TIME_COL}_denoised"] = _denoised.obs[
                    f"program_{_rapa_program}_time"
                ]

                if k[1] != "WT":
                    continue
                else:
                    _idx = self._all_data_expt_index(*k)

                    self.all_data.obs.loc[
                        _idx, f"{RAPA_TIME_COL}_denoised"
                    ] = _expt.obs[f"{RAPA_TIME_COL}_denoised"]
                    self.all_data.obs.loc[
                        _idx, f"{CC_TIME_COL}_denoised"
                    ] = _expt.obs[f"{CC_TIME_COL}_denoised"]

            self.save()

    def calc_gene_dists(self, recalculate=False):
        _program_ref = self.all_data.uns["programs"]
        _metrics = ["cosine", "euclidean", "manhattan", "information"]

        if recalculate or any(
            map(lambda x: f"{x}_distance" not in _program_ref, _metrics)
        ):
            _dists = _ad.AnnData(
                self.all_data.layers["counts"],
                var=self.all_data.var
            )

            _dists = self._normalize(
                _dists,
                method='scale',
                n_counts=NO_RP_DEPTH
            )

            _dists._inplace_subset_var(self.all_data.var["leiden"] != "-1")

            _sc.pp.pca(_dists, n_comps=_program_ref["n_comps"])

            print(f"Using {_program_ref['n_comps']} PCs for distance")
            _dists = (_dists.obsm["X_pca"] @ _dists.varm["PCs"].T).T

            _gc.collect()

            for metric in _metrics:
                print(f"Calculating metric {metric} on data {_dists.shape}")
                _metric_key = f"{metric}_distance"

                if recalculate or _metric_key not in _program_ref:
                    if metric != "information":
                        _program_ref[_metric_key] = pairwise_distances(
                            _dists, metric=metric, n_jobs=-1
                        )
                    else:
                        _program_ref[
                            _metric_key
                        ] = _ifv.metrics.information_distance(
                            _ifv.metrics.make_array_discrete(
                                _dists.T,
                                10,
                                axis=0
                            ),
                            10,
                            logtype=_np.log2,
                        )

            del _dists

            self.calc_gene_cosine_dists_all(recalculate=recalculate)

            self.save()

    def calc_gene_cosine_dists_all(self, recalculate=False):
        if recalculate or "cosine_distance" not in self.all_data.varp:
            print("Calculating cosine distance for all genes")

            _dists = _ad.AnnData(
                self.all_data.layers["counts"],
                var=self.all_data.var
            )

            _dists = self._normalize(
                _dists,
                method='scale',
                n_counts=NO_RP_DEPTH
            )

            self.all_data.varp["cosine_distance"] = pairwise_distances(
                _dists.X.T, metric="cosine", n_jobs=None
            )

    def _all_data_expt_index(self, expt, gene):
        _idx = self.all_data.obs["Experiment"] == expt
        _idx &= self.all_data.obs["Gene"] == gene
        return _idx

    def _transfer_programs(self, adata):
        adata.var["programs"] = self.all_data.var["programs"].reindex(
            adata.var_names
        )
        adata.obs["Pool_Combined"] = adata.obs["Pool"].astype(str)
        adata.obs.loc[
            adata.obs["Pool_Combined"] == "1",
            "Pool_Combined"
        ] = "12"
        adata.obs.loc[
            adata.obs["Pool_Combined"] == "2",
            "Pool_Combined"
        ] = "12"

        _pref = self.all_data.uns["programs"]
        adata.uns["programs"] = {
            "rapa_program": _pref["rapa_program"],
            "cell_cycle_program": _pref["cell_cycle_program"],
            "program_names": _pref["program_names"],
        }

        return adata


class CommonNames:
    _gene_names = None

    @classmethod
    def _load(cls):
        cls._gene_names = _pd.read_csv(
            GENE_NAMES_FILE,
            sep="\t",
            index_col=0
        )
        na_names = _pd.isna(cls._gene_names["Name"])

        # Replace empties with systematic names
        _replacers = cls._gene_names.index[na_names].tolist()
        cls._gene_names.loc[na_names, "Name"] = _replacers
        cls._gene_names = cls._gene_names.astype(str)

    @classmethod
    def common_name(cls, gene_symbol):
        if cls._gene_names is None:
            cls._load()

        if gene_symbol in cls._gene_names.index:
            return cls._gene_names.loc[gene_symbol].iloc[0]
        else:
            return gene_symbol


def common_name(gene):
    return CommonNames.common_name(gene)


def _gene_metadata(adata):
    # Gene common names
    if "CommonName" not in adata.var:
        if VERBOSE:
            print(f"Loading gene names from {GENE_NAMES_FILE}")
        yeast_gene_names = _pd.read_csv(GENE_NAMES_FILE, sep="\t", index_col=0)
        yeast_gene_names = yeast_gene_names.reindex(adata.var_names)
        na_names = _pd.isna(yeast_gene_names)
        yeast_gene_names[na_names] = adata.var_names.values[
            na_names.values.flatten()
        ].tolist()
        adata.var["CommonName"] = yeast_gene_names.astype(str)

    # Gene groups
    if "iESR" not in adata.var:
        if VERBOSE:
            print(f"Loading gene metadata from {GENE_GROUP_FILE}")
        ygg = _pd.read_csv(GENE_GROUP_FILE, sep="\t", index_col=0)
        ygg = (
            (ygg.pivot(columns="Group", values="Source").fillna(0) != 0)
            .reindex(adata.var_names)
            .fillna(False)
        )
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
            cc_prop[:, i] = (
                data.layers["counts"][:, data.var[cc].values].sum(axis=1).A1
                / data.obs["n_counts"].values
            )

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
            data.obs[ag] = (
                data.layers["counts"][:, data.var[ag].values]
                .sum(axis=1)
                .A1.astype(float)
                / data.obs["n_counts"].values
            )

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
    rapa_bulk_data.index.name = None
    rapa_bulk_metadata = (
        rapa_bulk_data[RAPA_BULK_EXPR_FILE_META_DATA_COLS]
        .astype(str)
        .astype("category")
    )
    rapa_bulk_data.drop(
        RAPA_BULK_EXPR_FILE_META_DATA_COLS,
        axis=1,
        inplace=True
    )
    return rapa_bulk_data, rapa_bulk_metadata


def rapa_bulk_times(include_0=False):
    rbt = (
        RAPA_BULK_EXPR_FILE_TIMES
        if not include_0 else
        [0] + RAPA_BULK_EXPR_FILE_TIMES
    )
    return _pd.Series(rbt).astype(str).tolist()


def sum_for_pseudobulk(adata, by_cols):
    group_bulk = []
    meta_bulk = []

    grouper = adata.obs.groupby(by_cols)

    for groups, group_data in grouper:
        idx = adata.obs_names.isin(group_data.index)
        _count_data = adata.layers["counts"][idx, :]

        try:
            _count_data = _count_data.A
        except AttributeError:
            pass

        group_bulk.append(
            _pd.DataFrame(
                _np.sum(_count_data, axis=0).reshape(1, -1),
                columns=adata.var_names
            )
        )

        meta_bulk.append(_pd.DataFrame([groups], columns=by_cols))

    group_bulk = _pd.concat(group_bulk)
    group_bulk.reset_index(drop=True, inplace=True)

    meta_bulk = _pd.concat(meta_bulk).astype("str").astype("category")
    meta_bulk.reset_index(drop=True, inplace=True)

    return group_bulk, meta_bulk


# SELECT 15 DCs FROM PALANTIR: #
def _select_palantir_dcs(df, n_dcs=15):
    col_split = list(map(
        lambda x: x.split("_"),
        df.columns.get_level_values(2)
    ))
    keep_cols = [int(x[1]) == n_dcs for x in col_split]
    keep_col_values = _pd.MultiIndex.from_tuples(
        [
            (df.columns[i][0], df.columns[i][1], str(x[0]) + "_" + str(x[2]))
            for i, (x, y) in enumerate(zip(col_split, keep_cols))
            if y
        ]
    )
    df = df.loc[:, keep_cols].copy()
    df.columns = keep_col_values
    return df


def _add_broad_category(adata):
    adata.var["category"] = "NA"
    adata.var.loc[
        functools.reduce(
            lambda x, y: x | adata.var[y],
            [adata.var["G1"], "G2", "M", "M-G1", "S"]
        ),
        "category",
    ] = "CC"
    adata.var.loc[
        functools.reduce(
            lambda x, y: adata.var[x] | adata.var[y],
            ["RP", "RiBi"]
        ),
        "category",
    ] = "RP"
    adata.var["category"] = _pd.Categorical(
        adata.var["category"], categories=["NA", "RP", "CC"]
    )
