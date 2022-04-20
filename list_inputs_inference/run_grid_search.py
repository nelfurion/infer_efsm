from .grid_search_setup import GSSetup
from .infer_vending_machine_grid_search import Estimator as VMEstimator
from .infer_vending_machine_grid_search import x_y_list as vm_inputs

gs_setup = GSSetup(VMEstimator(), 'vending_machine', vm_inputs)
gs_setup.run()