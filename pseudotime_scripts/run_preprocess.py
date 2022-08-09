import inferelator_velocity as ifv
import gc

from jtb_2022_code import FigureSingleCellData

data = FigureSingleCellData()
data.do_projections()
data.process_programs(recalculate=True)
data.calc_gene_dists()

for k in data.expts:
    data.decay_data(*k)
    print(f"GC: {gc.collect()}")
    
data.decay_data_all(recalculate=True)