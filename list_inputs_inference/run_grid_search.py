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

from list_inputs_inference.infer_odd_even import Estimator as OddEvenEstimator
from list_inputs_inference.infer_odd_even import x_y_list as odd_even_inputs
from list_inputs_inference.infer_odd_even import y_list as odd_even_outputs

from list_inputs_inference.infer_odd_even_multiobjective import Estimator as OddEvenMultiEstimator

if len(sys.argv) < 5:
  print("-------------------------------------------------------------------------------------------------------------------")
  print('Please provide the following parameters:')
  print('1 - Selection operator - one of: sel_tourn, sel_tourn_double, sel_best sel_stoch sel_lexicase sel_auto_eps_lexicase')
  print('2 - Iteration. Each iterations run the setup ONCE through 5-fold Shuffle split with 10% of the data as test size')
  print('3 - Function to use - one of [bmi_class, vm, odd_even]')
  print('4 - Function output type [str, float')
  print("-------------------------------------------------------------------------------------------------------------------")

  sys.exit()

setup = {
  'bmi_class': { 
    'estimator': GSSetup(BMIEstimator(), 'eaMuPlusLambda', 'bmi', bmi_inputs, bmi_outputs)
  },
  'vm': {
    'estimator': GSSetup(VMEstimator(), 'eaMuPlusLambda', 'vending_machine', vm_inputs, vm_outputs)
  },
  'odd_even': {
    'estimator': GSSetup(OddEvenEstimator(), 'eaMuPlusLambda', 'odd_even', odd_even_inputs, odd_even_outputs)
  },
  # 'odd_even_multi': {
  #   'estimator': GSSetup(OddEvenMultiEstimator(), 'eaMuPlusLambda', 'odd_even', odd_even_inputs, odd_even_outputs)
  # }
}

estimator = setup[sys.argv[3]]['estimator']
if __name__ == '__main__':
  print('SYS ARGV')
  print(sys.argv)
  estimator.run()
  print('Done')