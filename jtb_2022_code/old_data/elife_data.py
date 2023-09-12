import os
import pandas as pd
import anndata as ad
import numpy as np
import scipy.sparse
import scanpy as sc
import torch

from jtb_2022_code.utils.model_prediction import predict_all

from supirfactor_dynamical import (
    TimeDataset,
    TruncRobustScaler,
    predict_perturbation,
    perturbation_tfa_gradient,
)
from torch.utils.data import DataLoader

from ..figure_constants import (
    ELIFE_SINGLE_CELL_FILE,
    ELIFE_SINGLE_CELL_FILE_PROCESSED
)
from ..utils.figure_data import _gene_metadata, _call_cc

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
    "gcn4": "YEL009C",
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
            ).fillna(0.0).astype(np.float32)

        self.data = ad.AnnData(df, obs=obs_data, dtype=np.float32)

        self.data.layers["counts"] = scipy.sparse.csr_matrix(
            self.data.X.astype(np.int32)
        )

        if align_adata is not None and "programs" in align_adata.var:
            self.data.var["programs"] = align_adata.var["programs"]

        if align_adata is not None and "programs" in align_adata.uns:
            self.data.uns["programs"] = align_adata.uns["programs"]

        self.data.obs["n_counts"] = self.data.X.sum(axis=1).astype(int)
        self.data.obs["n_genes"] = self.data.X.astype(bool).sum(axis=1)

        self.data = _gene_metadata(self.data)
        self.data = _call_cc(self.data)

        print(f"Writing {ELIFE_SINGLE_CELL_FILE_PROCESSED}")
        self.data.write(ELIFE_SINGLE_CELL_FILE_PROCESSED)

    def get_data(
        self,
        genotype="WT",
        condition="YPD"
    ):
        return self.data[self._get_index(genotype, condition), :].copy()

    def get_pseudobulk(self, genotype="WT", condition="YPD"):
        if self.pseudobulk is None:
            self.pseudobulk = pd.DataFrame(
                self.data.layers["counts"].A,
                index=self.data.obs_names,
                columns=self.data.var_names,
            )

            self.pseudobulk[
                ["Genotype_Individual", "Condition"]
            ] = self.data.obs[
                ["Genotype_Individual", "Condition"]
            ]
            self.pseudobulk = self.pseudobulk.groupby(
                ["Genotype_Individual", "Condition"]
            ).agg("sum")

            meta_df = self.pseudobulk.index.to_frame().reset_index(drop=True)
            self.pseudobulk = self.pseudobulk.reset_index(drop=True)

            self.pseudobulk = ad.AnnData(self.pseudobulk, dtype=np.int32)

            self.pseudobulk.obs = meta_df
            self.pseudobulk.obs[["Genotype_Group", "Replicate"]] = meta_df[
                "Genotype_Individual"
            ].str.split("_", expand=True)
            self.pseudobulk.obs["n_counts"] = self.pseudobulk.X.sum(
                axis=1
            ).astype(int)

        return self.pseudobulk[
            self._get_index(genotype, condition, adata=self.pseudobulk),
            :
        ].copy()

    def _get_index(self, genotype, condition, adata=None):
        if adata is None:
            adata = self.data

        # Get genotype index
        if isinstance(genotype, (list, tuple)):
            _idx = adata.obs["Genotype_Group"].isin(
                ["WT(ho)" if g == "WT" else g for g in genotype]
            )

        elif genotype is not None:
            genotype = "WT(ho)" if genotype == "WT" else genotype
            _idx = adata.obs["Genotype_Group"] == genotype

        else:
            _idx = pd.Series(True, index=adata.obs_names)

        # Get genotype index
        if isinstance(condition, (list, tuple)):
            _idx &= adata.obs["Condition"].isin(condition)

        elif condition is not None:
            _idx &= adata.obs["Condition"] == condition

        return _idx


def get_elife_model_predictions(
    elife,
    biophysical_model,
):
    ypd, ypd_predicts, scaler = get_elife_predicts(
        elife,
        biophysical_model,
        None
    )
    ypd.X = ypd.X.A

    rapa, _ = get_elife_data(
        elife,
        genotype=None,
        rapa=True,
        scaler=scaler,
        genes=biophysical_model.prior_network_labels[0],
    )

    print("Predicting perturbations")
    with torch.no_grad():
        predictions = {
            v: np.multiply(
                predict_perturbation(
                    biophysical_model,
                    get_elife_tensor(
                        elife,
                        genotype=k if k is not None else "WT(ho)",
                        scaler=scaler,
                        genes=biophysical_model.prior_network_labels[0],
                    ),
                    v,
                    60,
                    unmodified_counts=True,
                )[1]
                .detach()
                .numpy(),
                biophysical_model._count_inverse_scaler.numpy()[None, None, :],
            )
            for k, v in TF_LOOKUP.items()
            if (v is None or v in biophysical_model.prior_network_labels[1])
        }

    print("Predicting TFA")
    with torch.no_grad():
        tfa_predictions = {
            v: predict_perturbation(
                biophysical_model,
                get_elife_tensor(
                    elife,
                    genotype=k if k is not None else "WT(ho)",
                    scaler=scaler,
                    genes=biophysical_model.prior_network_labels[0],
                ),
                v,
                60,
                unmodified_counts=True,
            )[3]
            .detach()
            .numpy()
            for k, v in TF_LOOKUP.items()
            if (v is None or v in biophysical_model.prior_network_labels[1])
        }

    def _get_matched_data(genotype, seed=50):
        if genotype is None:
            genotype = "WT(ho)"

        rng = np.random.default_rng(seed)

        _input = get_elife_tensor(
            elife,
            genotype=genotype,
            scaler=scaler,
            genes=biophysical_model.prior_network_labels[0],
        )

        _output_idx = rng.choice(
            np.arange(rapa.shape[0])[rapa.obs["Genotype_Group"] == genotype],
            size=_input.shape[0],
        )
        _output = torch.Tensor(
            np.expand_dims(rapa.layers["scaled"][_output_idx, :].A, 1)
        )

        return _input, _output

    print("Predicting TFA Marginal Error")
    prediction_gradients = {
        v: perturbation_tfa_gradient(
            biophysical_model,
            *_get_matched_data(k),
            perturbation=v,
            observed_data_delta_t=30,
        )
        for k, v in TF_LOOKUP.items()
        if (v is None or v in biophysical_model.prior_network_labels[1])
    }

    return (
        ypd,
        rapa,
        scaler,
        predictions,
        tfa_predictions,
        prediction_gradients
    )


def get_elife_data(
    elife,
    genotype="WT",
    scaler=None,
    rapa=False,
    genes=None
):
    data = elife.get_data(genotype, "RAPA" if rapa else "YPD")
    data.X = data.layers["counts"].astype(float)
    sc.pp.normalize_per_cell(data, counts_per_cell_after=3099)

    if genes is not None and (
        (len(genes) != data.shape[1]) or not all(data.var_names == genes)
    ):
        data = data[:, genes].copy()

    if rapa:
        data.obs["program_rapa_time"] = np.random.default_rng(100).uniform(
            27.5, 32.5, size=data.shape[0]
        )
    else:
        data.obs["program_rapa_time"] = np.random.default_rng(100).uniform(
            -10, 0, size=data.shape[0]
        )

    data.obs["Test"] = True

    if scaler is not None:
        data.layers["scaled"] = scaler.transform(data.X)
        data.var["scale"] = scaler.scale_
        data_scaler = scaler
    else:
        data_scaler = TruncRobustScaler(with_centering=False)
        data.layers["scaled"] = data_scaler.fit_transform(data.X)
        data.var["scale"] = data_scaler.scale_

    return data, data_scaler


def get_elife_tensor(*args, layer="scaled", **kwargs):
    data, _ = get_elife_data(*args, **kwargs)

    return next(
        iter(
            DataLoader(
                TimeDataset(
                    data.X if layer == "X" else data.layers[layer],
                    data.obs["program_rapa_time"],
                    -10,
                    0,
                    1,
                    sequence_length=10,
                    shuffle_time_vector=[-10, 0],
                ),
                batch_size=1000,
                drop_last=False,
            )
        )
    )


def get_elife_predicts(
    elife,
    model,
    genotype="WT",
    scaler=None
):
    ypd, ypd_scaler = get_elife_data(
        elife,
        genotype,
        genes=model.prior_network_labels[0],
        scaler=scaler
    )

    _, ypd_predicts = predict_all(ypd, data_processed=True)

    return ypd, ypd_predicts, ypd_scaler
