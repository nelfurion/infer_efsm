import datetime

from list_inputs_inference.grid_search_setup import GSSetup
from list_inputs_inference.infer_vending_machine_grid_search import Estimator as VMEstimator
from list_inputs_inference.infer_vending_machine_grid_search import x_y_list as vm_inputs
from list_inputs_inference.infer_vending_machine_grid_search import y_list as vm_outputs

import dill as pickle

import sys
sys.path.append("..")

futures_not_loaded = 'scoop.futures' not in sys.modules
controller_not_started = not (
    sys.modules['scoop.futures'].__dict__.get("_controller", None)
)

print("futures_not_loaded: ", futures_not_loaded)
print("controller_not_started: ", controller_not_started)


gs_setup = GSSetup(VMEstimator(), 'eaMuPlusLambda', 'vending_machine', vm_inputs, vm_outputs)
if __name__ == '__main__':
  gs_setup.run()

# if __name__ == '__main__':
  
#   # import sys
#   # print('*****************************', file=sys.stderr)
#   # print('*****************************', file=sys.stderr)
#   # print(str(self.tournparssize), file=sys.stderr)
#   # print('*****************************', file=sys.stderr)
#   # print('*****************************', file=sys.stderr)

#   print('DONE:')
#   print(str(datetime.datetime.now()))