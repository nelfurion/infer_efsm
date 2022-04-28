import sys, os
# sys.path.append("..")
sys.path.append(os.getcwd())

import datetime

from list_inputs_inference.grid_search_setup import GSSetup

from list_inputs_inference.infer_vending_machine_grid_search import Estimator as VMEstimator
from list_inputs_inference.infer_vending_machine_grid_search import x_y_list as vm_inputs
from list_inputs_inference.infer_vending_machine_grid_search import y_list as vm_outputs

from list_inputs_inference.infer_bmi import Estimator as BMIEstimator
from list_inputs_inference.infer_bmi import x_y_list as bmi_inputs
from list_inputs_inference.infer_bmi import y_list as bmi_outputs

if len(sys.argv) == 1:
  print('Please provide the following parameters:')
  print('1 - Selection operator - one of: sel_tourn, sel_tourn_double, sel_best sel_stoch sel_lexicase sel_auto_eps_lexicase')
  print('2 - Iteration. Each iterations run the setup ONCE through 5-fold Shuffle split with 10% of the data as test size')
  print('Please provide the following parameters:')

gs_setup_vm = GSSetup(VMEstimator(), 'eaMuPlusLambda', 'vending_machine', vm_inputs, vm_outputs)
gs_setup_bmi = GSSetup(BMIEstimator(), 'eaMuPlusLambda', 'bmi', bmi_inputs, bmi_outputs)
if __name__ == '__main__':
  print('SYS ARGV')
  print(sys.argv)
  gs_setup_bmi.run()
  gs_setup_vm.run()
  print('Done')