import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from jtb_2022_code.figure_constants import *

_labelsize = 6

_max_n = 12
cat_colors = [colors.rgb2hex(plt.get_cmap('plasma', _max_n)(c)) for c in range(_max_n)]

rng = np.random.default_rng(100)
rng.shuffle(cat_colors)

metric_label = {
    "AUPR": "AUPR",
    "R2_validation": "R${}^2$"
}

def load_model_results(modeling_type=None, trim_nans=True):

    supirfactor_results = pd.read_csv(MODEL_RESULTS_FILE, sep="\t", low_memory=False)
    supirfactor_results['Model_Type'] = supirfactor_results['Model_Type'].str.replace("_velocity", "")
    supirfactor_results['Shuffle'] = supirfactor_results['Shuffle'].astype(str)
    
    if modeling_type is not None:
        supirfactor_results = supirfactor_results[supirfactor_results[modeling_type]].copy()
    
    if trim_nans:
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
        supirfactor_results['Decay_Model_Width'] = supirfactor_results['Decay_Model_Width'].astype(
            pd.CategoricalDtype(
                np.sort(supirfactor_results['Decay_Model_Width'].unique()),
                ordered=True
            )
        )

    supirfactor_losses = pd.read_csv(MODEL_LOSSES_FILE, sep="\t", low_memory=False)
    supirfactor_losses['Model_Type'] = supirfactor_losses['Model_Type'].str.replace("_velocity", "")
    supirfactor_losses['Shuffle'] = supirfactor_losses['Shuffle'].astype(str)
    
    if modeling_type is not None:
        supirfactor_losses = supirfactor_losses[supirfactor_losses[modeling_type]].copy()

    if trim_nans:
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

        supirfactor_results['Decay_Model_Width'] = supirfactor_results['Decay_Model_Width'].astype(
            pd.CategoricalDtype(
                np.sort(supirfactor_results['Decay_Model_Width'].unique()),
                ordered=True
            )
        )

    return supirfactor_results, supirfactor_losses

def summarize_model_results(model_results, modeling_type='Counts'):
    
    _summary_idx = (model_results['Learning_Rate'] == 5e-5) & (model_results['Weight_Decay'] == 1e-7)
    _summary_idx |= pd.isna(model_results['Learning_Rate'])
    _summary_idx &= (model_results['Shuffle'] == "False") | (model_results['Shuffle'] == "Prior")
    
    summary_results = model_results[_summary_idx].copy()

    summary_results['model'] = pd.NA
    summary_results['prior_shuffle'] = summary_results['Shuffle'] == "Prior"

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

    summary_results = summary_results.dropna(subset='model').drop_duplicates(subset=['model', 'Seed', 'prior_shuffle'])

    summary_results['model'] = summary_results['model'].astype(pd.CategoricalDtype(['inferelator', 'static_meta', 'rnn', 'rnn_predictive', 'rnn_tuned'], ordered=True))
    summary_results['x_loc'] = summary_results['model'].cat.codes * 4 + summary_results['prior_shuffle'].astype(int) + 0.5
    summary_results['x_color'] = summary_results['model'].cat.codes.map(lambda x: colors.rgb2hex(matplotlib.colormaps['Dark2'](x)))
    summary_results.loc[summary_results['prior_shuffle'], 'x_color'] = "#000000"

    model_stats = summary_results.groupby(['x_loc', 'model', 'prior_shuffle', 'x_color'])['AUPR'].agg(['std', 'mean']).dropna().reset_index()
    
    return summary_results, model_stats

def plot_results(plot_data, metric, param, ax, ylim=(0, 0.3), yticks=(0.0, 0.1, 0.2, 0.3)):
    _n = len(plot_data[param].cat.categories)
    plot_data['x_loc'] = plot_data[param].cat.codes
    plot_data['x_color'] = plot_data[param].cat.codes.map(lambda x: cat_colors[x])

    plot_stats = plot_data.groupby(
        ['x_loc', 'x_color']
    )[metric].agg(
        ['std', 'mean']
    ).dropna().reset_index()

    ax.scatter(
        plot_data['x_loc'] + rng.uniform(-0.2, 0.2, plot_data.shape[0]),
        plot_data[metric],
        color=plot_data['x_color'],
        s=5,
        alpha=0.7
    )

    ax.scatter(
        plot_stats['x_loc'] + 0.5,
        plot_stats['mean'],
        color=plot_stats['x_color'],
        edgecolor='black',
        linewidth=0.25,
        s=15,
        alpha=1
    )

    ax.errorbar(
        plot_stats['x_loc'] + 0.5,
        plot_stats['mean'],
        yerr=plot_stats['std'],
        fmt='none',
        color='black',
        alpha=1,
        linewidth=0.5,
        zorder=-1
    )

    ax.set_ylim(*ylim)
    ax.set_yticks(yticks, yticks, size=_labelsize)
    ax.set_xlim(-0.5, _n)
    ax.set_xticks(
        [x + 0.25 for x in range(_n)], 
        [str(f"{x:.0e}") for x in plot_data[param].cat.categories],
        size=_labelsize,
        rotation=90
    )
    ax.set_title(metric_label[metric], size=8)
    
def plot_losses(plot_data, metric, param, ax, ylim=(0, 1.0), tuned=False):
    if tuned:
        metric = list(map(str, range(1, 401)))
        x_arange = np.arange(1, 401)
    else:
        x_arange = np.arange(1, 201)

    for x, cat in enumerate(plot_data[param].cat.categories):
        _cat_data = plot_data.loc[plot_data[param] == cat, metric]

        loss_color = cat_colors[x]

        if _cat_data.shape[0] > 0:
            ax.plot(
                x_arange,
                _cat_data.fillna(1e6).astype(float).values.T,
                color=cat_colors[x],
                alpha=0.1,
                zorder=rng.choice(
                    np.arange(1, _max_n) * -1,
                )
            )

    ax.tick_params(axis='both', which='major', labelsize=_labelsize)
    ax.set_ylim(*ylim)
    ax.set_yticks([0, ylim[1] / 2, ylim[1]], [0,ylim[1] / 2, ylim[1]])
    ax.set_xlabel("Epochs", size=8)
    ax.set_title("MSE", size=8)

def get_plot_idx(
    df,
    model,
    other_param=None,
    other_param_val=None,
    time='rapa',
    offset_time=10,
    model_type="Counts",
    loss_type="validation"
):
    plot_idx = df["Layer"] == "X"
    plot_idx &= df["Shuffle"] == "False"
    plot_idx &= df[model_type]
    
    if model == 'tuned':
        plot_idx &= df['Pretrained_Model'] == True
    else:
        plot_idx &= df['Pretrained_Model'] == False
        
    if model in ['tuned' ,'rnn_predictive']:
        plot_idx &= df['Output_Layer_Time_Offset'] == offset_time
    else:
        plot_idx &= df['Output_Layer_Time_Offset'] == 0
    
    plot_idx &= df['Time_Axis'] == time
        
    if (model == 'tuned') and (time == 'combined'):
        pass
    elif model == 'rnn_predictive':
        plot_idx &= df['Model_Type'] == 'rnn'
    else:
        plot_idx &= df['Model_Type'] == model
        
    if 'Loss_Type' in df.columns:
        plot_idx &= df['Loss_Type'] == loss_type
        
    if other_param is not None:
        plot_idx &= df[other_param] == other_param_val
        
    return plot_idx