from jtb_2022_code.utils.figure_filenames import parse_file_path_command_line


def main():

    parse_file_path_command_line()
    generate_figures()

def generate_figures():
    
    from jtb_2022_code.preprocess import preprocess

    from jtb_2022_code.figure_2 import plot_figure_2
    from jtb_2022_code.figure_2_supplemental import (
        figure_2_supplement_1_plot,
        figure_2_supplement_2_plot,
        figure_2_supplement_3_plot,
        figure_2_supplement_5_12_plot,
        figure_2_supplement_13_plot,
        figure_2_supplement_4_plot
    )

    from jtb_2022_code.figure_3 import figure_3_plot
    from jtb_2022_code.figure_3_supplemental import (
        figure_3_supplement_1_plot,
        figure_3_supplement_2_plot,
        figure_3_supplement_3_plot,
        figure_3_supplement_4_plot
    )
    
    data = preprocess()

    plot_figure_2(data)
    figure_2_supplement_1_plot(data)
    figure_2_supplement_2_plot(data)
    figure_2_supplement_3_plot(data.all_data)
    figure_2_supplement_4_plot(data)
    figure_2_supplement_5_12_plot(data)
    figure_2_supplement_13_plot(data)
    
    fig3, fig3_data = figure_3_plot(data)
    figure_3_supplement_1_plot(data)
    figure_3_supplement_2_plot(data)
    figure_3_supplement_4_plot(data, f3_data=fig3_data)

    
if __name__ == "__main__":
    main()
