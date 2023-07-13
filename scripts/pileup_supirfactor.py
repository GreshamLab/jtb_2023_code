import pandas as pd
import glob
import os

paths = {
    "Counts": "supirfactor_cell_cycle_search",
    "Velocity": "supirfactor_velocity_models",
    "Decay": "supirfactor_decay_models"
}

fill_cols = [
    ("Pretrained_Model", False, bool),
    ("Counts", False, bool),
    ("Velocity", False, bool),
    ("Decay", False, bool),
    ("Input_Dropout", 0.5, float),
    ("Hidden_Layer_Dropout", 0.0, float),
    ("Output_Layer_Time_Offset", 0, int)
]

drop_cols = ['Decay_Model', 'Static_Batch_Size', 'Dynamic_Batch_Size', 'Sequence_Length']

def _read(x, k):
    df = pd.read_csv(x, sep="\t")
    df['File_name'] = x
    df[k] = True
    return df

def _fix(x):
    for k, v, t in fill_cols:
        x.loc[pd.isna(x[k]), k] = v
        x[k] = x[k].astype(t)
    for col in drop_cols:
        del x[col]
    x['Model_Type'] = x['Model_Type'].str.replace("_velocity", "")
        
    return x    

for fn_in, fn_out in [
    ("*_RESULTS.tsv", "SUPIRFACTOR_RESULTS_ALL.tsv"),
    ("*_LOSSES.tsv", "SUPIRFACTOR_LOSSES_ALL.tsv")
]:
    _fix(
            pd.concat([
            _read(files, k)
            for k, path in paths.items()
            for files in glob.glob(os.path.join(path, fn_in))
        ])
    ).to_csv(
        fn_out,
        sep="\t",
        index=False
    )
