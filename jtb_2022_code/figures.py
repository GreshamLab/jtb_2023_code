import gc

from jtb_2022_code import FigureSingleCellData
from jtb_2022_code.utils.figure_data import calc_other_cc_groups

from jtb_2022_code.figure_1 import plot_figure_1
from jtb_2022_code.figure_1_supplemental import (
    figure_1_supplement_1_plot,
    figure_1_supplement_2_plot
)

from jtb_2022_code.figure_2 import plot_figure_2
from jtb_2022_code.figure_2_supplemental import (
    figure_2_supplement_1_plot,
    figure_2_supplement_2_plot,
    figure_2_supplement_3_plot,
    figure_2_supplement_4_11_plot,
    figure_2_supplement_12_plot
)

from jtb_2022_code.figure_3 import figure_3_plot
from jtb_2022_code.figure_3_supplemental import (
    figure_3_supplement_1_plot,
    figure_3_supplement_2_plot,
    figure_3_supplement_3_plot
)

from jtb_2022_code.figure_4 import figure_4_plot


def main():
    figures()


def figures():
    data = FigureSingleCellData()
    data.do_projections()
    
    plot_figure_1(data)
    
    data.apply_inplace_to_everything(calc_other_cc_groups)

    figure_1_supplement_1_plot(data)
    figure_1_supplement_2_plot(data)

    data.calc_gene_dists()
    data.process_programs()
    data.process_times()

    plot_figure_2(data)
    
    figure_2_supplement_1_plot(data)
    figure_2_supplement_2_plot(data)
    figure_2_supplement_3_plot(data.all_data)
    figure_2_supplement_4_11_plot(
        data,
        cc_program=data.all_data.uns['programs']['cell_cycle_program'],
        rapa_program=data.all_data.uns['programs']['rapa_program']
    )
    figure_2_supplement_12_plot(data)
    
    for k in data.expts:
        data.decay_data(*k)
        gc.collect()

    data.decay_data_all(recalculate)
    
    _, fig3_data = figure_3_plot(data)
    
    figure_3_supplement_1_plot(data)
    figure_3_supplement_2_plot(data)
    figure_3_supplement_3_plot(data, f3_data=fig3_data)
    
    plot_figure_4(data)
    
if __name__ == "__main__":
    main()
