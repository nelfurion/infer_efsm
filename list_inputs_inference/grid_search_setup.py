import copy
import sys
from pathlib import Path
from xml.etree.ElementTree import TreeBuilder
import pandas as pd
import multiprocessing
from params_grid import common_params_grid

from sklearn.model_selection import ShuffleSplit, GridSearchCV

class GSSetup():
  def __init__(self, estimator, gp_alg_name, function_name, function_inputs, function_outputs) -> None:
    self.function_inputs = function_inputs
    self.function_outputs = function_outputs
    self.estimator = estimator
    self.function_name = function_name
    self.results_function_dir = './results/' + gp_alg_name + '/' + self.function_name + '/'

  def run(self):
    # TOURNAMENT SELECTION

    selection = sys.argv[1]
    dir =  self.results_function_dir + selection + '/'
    Path(dir).mkdir(parents=True, exist_ok=True)

    params_grid = copy.deepcopy(common_params_grid)
    params_grid['tree_output_dir'] = [dir]
    params_grid['selection'] = [selection]

    self._run_grid_search_cv_and_log(dir, self.function_inputs, self.function_outputs, params_grid)

    print('***************************', file=sys.stderr)
    print(str(selection), " SELECTION DONE", file=sys.stderr)
    print('***************************', file=sys.stderr)

  def _run_grid_search_cv_and_log(self, dir, inputs, outputs, params_grid):
    grid_search = GridSearchCV(
      cv=ShuffleSplit(n_splits=1, test_size=0.1, random_state=7),
      n_jobs=multiprocessing.cpu_count(), #use all available processors
      pre_dispatch= multiprocessing.cpu_count() * 1.5, # Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process.
      error_score='raise',
      estimator=self.estimator, 
      verbose=10,
      param_grid=params_grid,
      refit=False
    )

    grid_search.fit(inputs, outputs)
    dataframe = pd.DataFrame(grid_search.cv_results_)

    dataframe.to_csv(dir + 'result' + '_' + sys.argv[2] + '.csv')
