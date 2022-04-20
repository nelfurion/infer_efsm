import copy
import sys
from pathlib import Path
from xml.etree.ElementTree import TreeBuilder
import pandas as pd

from sklearn.model_selection import GridSearchCV

class GSSetup():
  def __init__(self, estimator, function_name, function_inputs) -> None:
    self.function_inputs = function_inputs
    self.estimator = estimator
    self.function_name = function_name
    self.results_function_dir = './results/' + self.function_name + '/'

  def run(self):
    old_stdout = sys.stdout

    #  DCD TOURNAMENT SELECTION

    # selection = 'sel_tourn_dcd'
    # for i in range(26,27):
    #   dir = './results/vending_machine/' + selection + '/' + str(i) + '/'
    #   Path(dir).mkdir(parents=True, exist_ok=True)

    #   grid_search = GridSearchCV(
    #     estimator=Estimator(dir), 
    #     verbose=10,
    #     param_grid={
    #       'mu': [4, 8],
    #       'lmbda': [12], #[10, 20],
    #       'cxpb': [0.1], #[0.1, 0.2],
    #       'mutpb': [0.1], # [0.1, 0.2],
    #       'gcount': [50],
    #       'popsize': [100],
    #       'selection': [selection],
    #     },
    #   )

    #   log_file = open(dir + 'log.txt',"w")
    #   sys.stdout = log_file

    #   grid_search.fit(x_y_list, x_y_list)
    #   dataframe = pd.DataFrame(grid_search.cv_results_)
    #   dataframe.to_csv(dir + 'result' + '_' + str(i) + '.csv')

    #   log_file.close()



    common_params_grid = {
      'mu': [5, 10],
      'lmbda': [10], #[10, 20],
      'cxpb': [0.1], #[0.1, 0.2],
      'mutpb': [0.1], # [0.1, 0.2],
      'gcount': [50],
      'popsize': [100],
      'cx_tool': ['cxOnePoint', 'cxOnePointLeafBiased', 'cxSemantic'],
      'mut_tool': ['mutShrink', 'mutUniform', 'mutNodeReplacement', 'mutInsert', 'mutSemantic']
    }


    # TOURNAMENT SELECTION


    selection = 'sel_tourn'
    for i in range(26,27):
      dir =  self.results_function_dir + selection + '/' + str(i) + '/'
      Path(dir).mkdir(parents=True, exist_ok=True)

      params_grid = copy.deepcopy(common_params_grid)
      params_grid['tournsize'] = [2] # [2, 4, 7],
      params_grid['selection'] = [selection]
      params_grid['tree_output_dir'] = [dir]

      self._run_grid_search_cv_and_log(dir, self.function_inputs, params_grid, i)



    #  DOUBLE TOURNAMENT SELECTION


    selection = 'sel_tourn_double'
    for i in range(26,27):
      dir =  self.results_function_dir + selection + '/' + str(i) + '/'
      Path(dir).mkdir(parents=True, exist_ok=True)

      params_grid = copy.deepcopy(common_params_grid)
      params_grid['tournsize'] = [2] # [2, 4, 7],
      params_grid['tournparssize'] = [1.1] #[1.1, 1.4, 1.7],
      params_grid['selection'] = [selection]
      params_grid['tree_output_dir'] = [dir]

      print('*****************************', file=sys.stderr)
      print('*****************************', file=sys.stderr)
      print(str(params_grid), file=sys.stderr)
      print('*****************************', file=sys.stderr)
      print('*****************************', file=sys.stderr)

      self._run_grid_search_cv_and_log(dir, self.function_inputs, params_grid, i)


    #  RANDOM SELECTION

    selection_operators = ['sel_random', 'sel_best', 'sel_worst', 'sel_stoch', 'sel_lexicase', 'sel_eps_lexicase', 'sel_auto_eps_lexicase']
    for selection in selection_operators:

      for i in range(26,27):
        dir =  self.results_function_dir + selection + '/' + str(i) + '/'
        Path(dir).mkdir(parents=True, exist_ok=True)

        params_grid = copy.deepcopy(common_params_grid)
        params_grid['selection'] = [selection]
        params_grid['tree_output_dir'] = [dir]

        self._run_grid_search_cv_and_log(dir, self.function_inputs, params_grid, i)






    sys.stdout = old_stdout

  def _run_grid_search_cv_and_log(self, dir, inputs, params_grid, iteration):
    global treedir
    treedir = dir

    grid_search = GridSearchCV(
      error_score='raise',
      estimator=self.estimator, 
      verbose=10,
      param_grid=params_grid
    )

    log_file = open(dir + 'log.txt',"w")
    sys.stdout = log_file

    grid_search.fit(inputs, inputs)
    dataframe = pd.DataFrame(grid_search.cv_results_)
    dataframe.to_csv(dir + 'result' + '_' + str(iteration) + '.csv')

    log_file.close()
