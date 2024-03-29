import pandas as pd

from ..figure_constants import (
    DECAY_CONSTANT_FILES,
    DECAY_CONSTANT_LINKS,
    DataFile
)


def process_all_decay_links(genes):
    return pd.concat(
        [
            _process_link(genes, x)
            for x in DECAY_CONSTANT_LINKS.keys()
        ], axis=1
    )


def _process_link(genes, dataset):
    file_type, gene_col, hl_col, _ = DECAY_CONSTANT_FILES[dataset]

    f = str(DataFile(f"{dataset}.{file_type}"))
    df = pd.read_csv(f, sep="\t", index_col=0 if gene_col == "X1" else None)

    df, hl_col = _process_df_hl(df, genes, gene_col, hl_col)
    df.rename({hl_col: dataset}, axis=1, inplace=True)

    return df[[dataset]]


def _process_df_hl(df, genes, gene_col, hl_col):
    if gene_col == "X1":
        df.index.name = "X1"
        df.reset_index(inplace=True)

    if isinstance(hl_col, list):
        df["means"] = df[hl_col].mean(axis=1)
        hl_col = "means"

    df = df[[gene_col, hl_col]].groupby(gene_col).agg("mean")
    df = df.reindex(genes, axis=0)

    return df, hl_col
