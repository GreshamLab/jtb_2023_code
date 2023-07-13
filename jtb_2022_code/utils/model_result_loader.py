import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from jtb_2022_code.figure_constants import *

def load_model_results(modeling_type=None):

    supirfactor_results = pd.read_csv(str(DataFile("SUPIRFACTOR_RESULTS.tsv")), sep="\t")
    supirfactor_results['Model_Type'] = supirfactor_results['Model_Type'].str.replace("_velocity", "")
    
    if modeling_type is not None:
        supirfactor_results = supirfactor_results[supirfactor_results[modeling_type]].copy()
        
    supirfactor_results = supirfactor_results.dropna(subset='Learning_Rate')

    supirfactor_results['Learning_Rate'] = supirfactor_results['Learning_Rate'].astype(
        pd.CategoricalDtype(
            np.sort(supirfactor_results['Learning_Rate'].unique()),
            ordered=True
        )
    )
    supirfactor_results['Weight_Decay'] = supirfactor_results['Weight_Decay'].astype(
        pd.CategoricalDtype(
            np.sort(supirfactor_results['Weight_Decay'].unique()),
            ordered=True
        )
    )

    supirfactor_losses = pd.read_csv(str(DataFile("SUPIRFACTOR_LOSSES.tsv")), sep="\t")
    supirfactor_losses['Model_Type'] = supirfactor_losses['Model_Type'].str.replace("_velocity", "")
    
    if modeling_type is not None:
        supirfactor_losses = supirfactor_losses[supirfactor_losses[modeling_type]].copy()

    supirfactor_losses = supirfactor_losses.dropna(subset='Learning_Rate')

    supirfactor_losses['Learning_Rate'] = supirfactor_losses['Learning_Rate'].astype(
        pd.CategoricalDtype(
            np.sort(supirfactor_losses['Learning_Rate'].unique()),
            ordered=True
        )
    )

    supirfactor_losses['Weight_Decay'] = supirfactor_losses['Weight_Decay'].astype(
        pd.CategoricalDtype(
            np.sort(supirfactor_results['Weight_Decay'].unique()),
            ordered=True
        )
    )
    
    return supirfactor_results, supirfactor_losses

def summarize_model_results(model_results, modeling_type='Counts'):
    
    summary_results = model_results[
        (((model_results['Learning_Rate'] == 5e-5) &
        (model_results['Weight_Decay'] == 1e-7)) |
        pd.isna(model_results['Learning_Rate']))
    ].copy()

    summary_results['model'] = pd.NA
    summary_results['Shuffle'] = False

    summary_results.loc[
        summary_results['Model_Type'] == 'inferelator',
        'model'
    ] = 'inferelator'

    summary_results.loc[
        (summary_results['Model_Type'] == 'static_meta') &
        (summary_results['Time_Axis'] == 'rapa') &
        summary_results[modeling_type],
        'model'
    ] = 'static_meta'

    summary_results.loc[
        (summary_results['Output_Layer_Time_Offset'] == 0) &
        (summary_results['Time_Axis'] == 'combined') &
        ~summary_results['Pretrained_Model'] &
        summary_results[modeling_type],
        'model'
    ] = 'rnn'

    summary_results.loc[
        (summary_results['Output_Layer_Time_Offset'] == 10) &
        (summary_results['Time_Axis'] == 'combined') &
        ~summary_results['Pretrained_Model'] &
        summary_results[modeling_type],
        'model'
    ] = 'rnn_predictive'

    summary_results.loc[
        (summary_results['Output_Layer_Time_Offset'] == 10) &
        (summary_results['Time_Axis'] == 'combined') &
        summary_results['Pretrained_Model'] &
        summary_results[modeling_type],
        'model'
    ] = 'rnn_tuned'

    summary_results = summary_results.dropna(subset='model').drop_duplicates(subset=['model', 'Seed'])

    summary_results['model'] = summary_results['model'].astype(pd.CategoricalDtype(['inferelator', 'static_meta', 'rnn', 'rnn_predictive', 'rnn_tuned'], ordered=True))
    summary_results['x_loc'] = summary_results['model'].cat.codes * 4 + summary_results['Shuffle'].astype(int) + 0.5
    summary_results['x_color'] = summary_results['model'].cat.codes.map(lambda x: colors.rgb2hex(matplotlib.colormaps['Dark2'](x)))
    summary_results.loc[summary_results['Shuffle'], 'x_color'] = "#000000"

    model_stats = summary_results.groupby(['x_loc', 'model', 'Shuffle', 'x_color'])['AUPR'].agg(['std', 'mean']).dropna().reset_index()
    
    return summary_results, model_stats
