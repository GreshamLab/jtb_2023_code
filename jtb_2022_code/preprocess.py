import gc

from jtb_2022_code import FigureSingleCellData


def main():
    preprocess()


def preprocess():
    data = FigureSingleCellData()
    data.do_projections()
    data.process_programs()
    data.calc_gene_dists()

    for k in data.expts:
        data.decay_data(*k)
        gc.collect()

    data.decay_data_all(recalculate=True)


if __name__ == "__main__":
    main()
