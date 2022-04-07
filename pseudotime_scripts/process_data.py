from jtb_2022_code.utils.pseudotime_common import do_pca_pt, do_denoised_pca, calculate_times_velocities, do_time_assign_by_pool
from jtb_2022_code.utils.dewakss_common import run_dewakss
from jtb_2022_code.utils.decay_common import calc_decays, calc_halflives
from jtb_2022_code import FigureSingleCellData
from jtb_2022_code.utils.figure_data import calc_other_cc_groups
from jtb_2022_code.pseudotime.pseudotime_cellrank_dewakss import do_cytotrace_denoised
from jtb_2022_code.pseudotime.pseudotime_scanpy_dpt_dewakss import do_dpt_denoised
import numpy as np

data = FigureSingleCellData(start_from_scratch=True)

data.do_projections()
data.apply_inplace_to_everything(calc_other_cc_groups)
data.apply_inplace_to_everything(do_pca_pt)
data.apply_inplace_to_expts(run_dewakss)
data.save()

data.apply_inplace_to_expts(do_denoised_pca)
data.apply_inplace_to_expts(do_pca_pt, pca_key="denoised_pca", pt_key="denoised_pca_pt")
data.apply_inplace_to_expts(do_cytotrace_denoised)
data.apply_inplace_to_expts(do_dpt_denoised)
data.save()

data.apply_inplace_to_expts(do_time_assign_by_pool, 'pca_pt')
data.apply_inplace_to_expts(calculate_times_velocities, 
                            layer='denoised',
                            transform_expr=np.expm1,
                            distance_key='denoised_distances')
data.save()

data.apply_inplace_to_expts(calc_decays)
data.apply_inplace_to_expts(calc_halflives)
#data.apply_inplace_to_expts(calc_decay_windows, 80)
data.save()