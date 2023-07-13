import gc

from jtb_2022_code.utils.figure_filenames import parse_file_path_command_line


def main():

    parse_file_path_command_line()
    preprocess()


def preprocess():
    
    from jtb_2022_code.utils.figure_data import (
        FigureSingleCellData,
        calc_other_cc_groups
    )
    
    data = FigureSingleCellData()
   
    data.do_projections()
    data.apply_inplace_to_everything(calc_other_cc_groups)

    data.process_programs()

    for k in data.expts:
        data.decay_data(*k)
        gc.collect()

    data.decay_data_all()
    data.calc_gene_dists()
    
    return data


if __name__ == "__main__":
    main()
