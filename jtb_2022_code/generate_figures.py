import scanpy as sc
from jtb_2022_code.utils.figure_filenames import parse_file_path_command_line


def main():
    parse_file_path_command_line()
    generate_figures()


def generate_figures():
    from jtb_2022_code.preprocess import preprocess

    from jtb_2022_code.utils.model_prediction import (
        process_velocity_for_model,
        predict_all,
    )

    from jtb_2022_code.figure_constants import (
        SUPIRFACTOR_BIOPHYSICAL_MODEL
    )

    from jtb_2022_code.utils.model_result_loader import load_model_results

    from jtb_2022_code.old_data.elife_data import (
        OldElifeData,
        get_elife_model_predictions,
    )

    from supirfactor_dynamical import read

    from jtb_2022_code.figure_1 import plot_figure_1
    from jtb_2022_code.figure_1_supplemental import (
        figure_1_supplement_1_plot,
        figure_1_supplement_2_plot,
    )

    from jtb_2022_code.figure_2 import plot_figure_2
    from jtb_2022_code.figure_2_supplemental import (
        figure_2_supplement_1_plot,
        figure_2_supplement_2_plot,
        figure_2_supplement_3_plot,
        figure_2_supplement_5_12_plot,
        figure_2_supplement_13_plot,
        figure_2_supplement_4_plot,
        figure_2_supplement_14_plot,
    )

    from jtb_2022_code.figure_3 import figure_3_plot
    from jtb_2022_code.figure_3_supplemental import (
        figure_3_supplement_1_plot,
        figure_3_supplement_2_plot,
        figure_3_supplement_3_plot,
    )

    from jtb_2022_code.figure_4 import plot_figure_4
    from jtb_2022_code.figure_4_supplemental import (
        figure_4_supplement_1_plot,
        figure_4_supplement_2_plot,
        figure_4_supplement_3_plot,
        figure_4_supplement_4_plot,
    )

    from jtb_2022_code.figure_5 import plot_figure_5
    from jtb_2022_code.figure_5_supplemental import (
        figure_5_supplement_1_plot,
        figure_5_supplement_2_plot,
        figure_5_supplement_3_plot,
    )

    from jtb_2022_code.figure_6 import plot_figure_6

    from jtb_2022_code.figure_6_supplemental import figure_6_supplement_1_plot

    data = preprocess()

    plot_figure_1(data)
    figure_1_supplement_1_plot(data)
    figure_1_supplement_2_plot(data)

    plot_figure_2(data)
    figure_2_supplement_1_plot(data)
    figure_2_supplement_2_plot(data)
    figure_2_supplement_3_plot(data.all_data)
    figure_2_supplement_4_plot(data)
    figure_2_supplement_5_12_plot(data)
    figure_2_supplement_13_plot(data)
    figure_2_supplement_14_plot()

    supirfactor_results, supirfactor_losses = load_model_results()
    model_data, predicts, model_scaler = predict_all(data.all_data)
    predicts.X = predicts.layers["count_predict_counts"]

    figure_3_plot(model_data, predicts)
    figure_3_supplement_1_plot()
    figure_3_supplement_2_plot(supirfactor_results, supirfactor_losses)
    figure_3_supplement_3_plot(supirfactor_results, supirfactor_losses)

    # Do PCA for fig4
    predicts.X = predicts.layers["velocity_predict_counts"]
    sc.pp.pca(predicts, n_comps=2)
    sc.pp.pca(model_data, n_comps=2)

    # Process velocity data for fig4
    velo_data, velo_scaler = process_velocity_for_model(
        data,
        model_data.var_names
    )

    plot_figure_4(model_data, velo_data, predicts)
    figure_4_supplement_1_plot(data)
    figure_4_supplement_2_plot(data)
    figure_4_supplement_3_plot()
    figure_4_supplement_4_plot()

    plot_figure_5(model_data, velo_data, predicts)
    figure_5_supplement_1_plot(data)
    figure_5_supplement_2_plot(data)
    figure_5_supplement_3_plot(data, model_data, model_scaler)

    elife = OldElifeData()
    biophysical_model = read(SUPIRFACTOR_BIOPHYSICAL_MODEL).eval()

    (
        ypd,
        rapa,
        ypd_scaler,
        predictions,
        tfa_predictions,
        prediction_gradients,
    ) = get_elife_model_predictions(elife, biophysical_model)

    plot_figure_6(
        predictions,
        rapa,
        tfa_predictions,
        prediction_gradients,
        biophysical_model
    )

    figure_6_supplement_1_plot(predictions, rapa)


if __name__ == "__main__":
    main()
