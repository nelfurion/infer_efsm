import copy
import sys
from pathlib import Path
from xml.etree.ElementTree import TreeBuilder
import pandas as pd
import multiprocessing

from sklearn.model_selection import ShuffleSplit, GridSearchCV

class GSSetup():
  def __init__(self, estimator, gp_alg_name, function_name, function_inputs, function_outputs) -> None:
    self.function_inputs = function_inputs
    self.function_outputs = function_outputs
    self.estimator = estimator
    self.function_name = function_name
    self.results_function_dir = './results/' + gp_alg_name + '/' + self.function_name + '/'

  def run(self):
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

    # common_params_grid = {
    #   'mu': [5, 10, 100], #[5, 10, 100],
    #   'lmbda': [10, 50, 100], #[10, 50, 100], #[10, 20],
    #   'cxpb': [0.1, 0.5, 0.8], #[0.1, 0.5, 0.8], #[0.1, 0.2],
    #   'mutpb': [0.1, 0.15, 0.2], #[0.1, 0.15, 0.2], # [0.1, 0.2],
    #   'gcount': [50, 500, 1000],#[50, 1000],
    #   'popsize': [500, 1000, 10000],#[100, 1000, 10000],
    #   'cx_tool': ['cxOnePoint', 'cxOnePointLeafBiased', 'cxSemantic'],
    #   'mut_tool': ['mutShrink', 'mutUniform', 'mutNodeReplacement', 'mutInsert', 'mutSemantic'],
    #   'tournsize': [7],
    #   'tournparssize': [1.4],
    # }


    common_params_grid = {
      'mu': [100, 500, 1000, 1500, 2000], #[5, 10, 100],
      'lmbda': [10], #[10, 50, 100], #[10, 20],
      'cxpb': [0.8], #[0.1, 0.5, 0.8], #[0.1, 0.2],
      'mutpb': [0.15], #[0.1, 0.15, 0.2], # [0.1, 0.2],
      'gcount': [500],#[50, 1000],
      'popsize': [1000, 10000],#[100, 1000, 10000],
      'cx_tool': ['cxOnePoint'],
      'mut_tool': ['mutShrink'],
      'tournsize': [7],
      'tournparssize': [1.4],
    }

    # TOURNAMENT SELECTION


    # selection = 'sel_tourn'
    selection = sys.argv[1]
    dir =  self.results_function_dir + selection + '/'
    Path(dir).mkdir(parents=True, exist_ok=True)

    params_grid = copy.deepcopy(common_params_grid)
    params_grid['tree_output_dir'] = [dir]
    params_grid['selection'] = [selection]

    self._run_grid_search_cv_and_log(dir, self.function_inputs, self.function_outputs, params_grid)

    print('***************************', file=sys.stderr)
    print('***************************', file=sys.stderr)
    print('***************************', file=sys.stderr)
    print(str(selection), " SELECTION DONE", file=sys.stderr)
    print('***************************', file=sys.stderr)
    print('***************************', file=sys.stderr)
    print('***************************', file=sys.stderr)

    # DOUBLE TOURNAMENT SELECTION


    # selection = 'sel_tourn_double'
    # dir =  self.results_function_dir + selection + '/'
    # Path(dir).mkdir(parents=True, exist_ok=True)

    # params_grid = copy.deepcopy(common_params_grid)

    # params_grid['selection'] = [selection]
    # params_grid['tree_output_dir'] = [dir]

    # self._run_grid_search_cv_and_log(dir, self.function_inputs, self.function_outputs, params_grid)


    # print('***************************', file=sys.stderr)
    # print('***************************', file=sys.stderr)
    # print('***************************', file=sys.stderr)
    # print("DOUBLE SELECTION DONE", file=sys.stderr)
    # print('***************************', file=sys.stderr)
    # print('***************************', file=sys.stderr)
    # print('***************************', file=sys.stderr)

    # #  RANDOM SELECTION

    # selection_operators = ['sel_best', 'sel_stoch', 'sel_lexicase', 'sel_auto_eps_lexicase']
    # for selection in selection_operators:
    #   dir =  self.results_function_dir + selection + '/'
    #   Path(dir).mkdir(parents=True, exist_ok=True)

    #   params_grid = copy.deepcopy(common_params_grid)
    #   params_grid['selection'] = [selection]
    #   params_grid['tree_output_dir'] = [dir]

    #   self._run_grid_search_cv_and_log(dir, self.function_inputs, self.function_outputs, params_grid)

    #   print('***************************', file=sys.stderr)
    #   print('***************************', file=sys.stderr)
    #   print('***************************', file=sys.stderr)
    #   print(selection + " DONE", file=sys.stderr)
    #   print('***************************', file=sys.stderr)
    #   print('***************************', file=sys.stderr)
    #   print('***************************', file=sys.stderr)

  def _run_grid_search_cv_and_log(self, dir, inputs, outputs, params_grid):
    grid_search = GridSearchCV(
      cv=ShuffleSplit(n_splits=100, test_size=0.1, random_state=7),
      n_jobs=multiprocessing.cpu_count(), #use all available processors
      pre_dispatch= multiprocessing.cpu_count() * 1.5, # Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process.
      error_score='raise',
      estimator=self.estimator, 
      verbose=10,
      param_grid=params_grid
    )

    grid_search.fit(inputs, outputs)
    dataframe = pd.DataFrame(grid_search.cv_results_)

    dataframe.to_csv(dir + 'result' + '_' + sys.argv[2] + '.csv')
