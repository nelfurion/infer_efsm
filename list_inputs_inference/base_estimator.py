from deap import algorithms

from gp_algorithm import GPListInputAlgorithm
from lib import generate_random_string
from plot import plot_tree
import uuid
import pandas as pd
import numbers
import sys
import operator
from multiprocessing import Lock

mutex = Lock()

class BaseEstimator():
  def fit(self, target_x_y, y):
    self.setup['target'] = target_x_y
    self.setup['population_generation_func'] = lambda population, gpa: algorithms.eaMuPlusLambda(
      population,
      gpa.toolbox,
      self.mu,
      self.lmbda,
      self.cxpb,
      self.mutpb,
      gpa.generations_count,
      stats=gpa.mstats,
      halloffame=gpa.hof,
      verbose=True
    )

    self.gpa = GPListInputAlgorithm.create(self.setup)
    self.gpa.run()

    self.estimator = self.gpa.get_best_tree()

    print('BEST EST: ', self.estimator)

    best_tree_stats_string = self.gpa.mstats.get_best_generation_stats_string(self.estimator)

    # for some reason the GSCV is fit one more time after all runs are done, and it does not have the tree_output_dir param
    if self.tree_output_dir:
      with mutex:
        with open(self.tree_output_dir + 'best_trees_' + sys.argv[2] + '.csv', 'a') as best_trees_csv:
          best_trees_csv.write(str(self.get_params()) + ',' + best_tree_stats_string)
          best_trees_csv.write('\n')

    return self

  def set_params(self, mu, lmbda, cxpb, mutpb, gcount, popsize, mut_tool, cx_tool, selection, tree_output_dir, tournsize=None, tournparssize=None, fitness_weights=None, output_type=float):
    self.tree_output_dir = tree_output_dir
    self.mu = mu
    self.lmbda = lmbda
    self.cxpb = cxpb
    self.mutpb = mutpb
    self.gcount = gcount
    self.popsize = popsize
    self.mut_tool = mut_tool
    self.cx_tool = cx_tool
    self.selection = selection
    self.tournsize = tournsize
    self.tournparssize = tournparssize
    self.fitness_weights = fitness_weights
    self.output_type = output_type

    self.setup = {
      'population_size': popsize,
      'hall_of_fame_size': 2,
      'input_list_length': 1, # hardcoding it to only accept a single argument # event_args_length,
      'output_type': output_type,
      'generations_count': gcount,
      'primitives': [
        # [safe_binary_operation(operator.add, 0), [float, float], float, 'add'],
        # [safe_binary_operation(operator.sub, 0), [float, float], float, 'sub'],
        # [safe_binary_operation(operator.mul, 0), [float, float], float, 'mul'],
        # [protectedDivision, [float, float], float, 'div']
        [lambda first, second: int(round(first)) % int(round(second)), [numbers.Complex, numbers.Complex], int, 'mod'],
        [operator.add, [float, float], float, 'add'],
        [operator.sub, [float, float], float, 'sub'],
        [operator.mul, [numbers.Complex, numbers.Complex], float, 'mul'],
        [operator.truediv, [numbers.Complex, numbers.Complex], float, 'div'],

        [operator.ge, [float, float], bool, 'ge'],
        [operator.gt, [numbers.Complex, numbers.Complex], bool, 'gt'],
        [operator.le, [float, float], bool, 'le'],
        [operator.lt, [numbers.Complex, numbers.Complex], bool, 'lt'],
        [operator.eq, [float, float], bool, 'eq'],
        [operator.not_, [bool], bool, 'my_not'],
        [lambda bool, str_1, str_2: str_1 if bool else str_2 , [bool, str, str], str, 'str_if_else'],
      ],
      'terminals':[
        [1, float],
        [0, float],
        [2, float],
        [100, float],
        [18.5, float],
        [24.9, float],
        [True, bool],
        [False, bool],
        ['healthy', str],
        ['underweight', str],
        ['overweight', str],
        ['no', str],
        ['yes', str]
      ],
      'individual_fitness_eval_func': self.fitness_eval_fun,
      'mut_tool': mut_tool,
      'cx_tool': cx_tool,
      'selection': selection,
      'tournsize': tournsize,
      'tournparssize': tournparssize,
      'fitness_weights': fitness_weights
    }

    self.estimator = None
    self.gpa = None

    return self

  def get_params(self, deep=False):
    params = {
      'mu': self.mu or 'N/A',
      'lmbda': self.lmbda or 'N/A',
      'cxpb': self.cxpb or 'N/A',
      'mutpb': self.mutpb or 'N/A',
      'gcount': self.gcount or 'N/A',
      'popsize': self.popsize or 'N/A',
      'mut_tool': self.mut_tool or 'N/A',
      'cx_tool': self.cx_tool or 'N/A',
      'selection': self.selection or 'N/A',
      'tournsize': self.tournsize or 'N/A',
      'tournparssize': self.tournparssize or 'N/A',
      'fitness_weights': self.fitness_weights or 'N/A',
      'output_type': self.output_type or 'N/A'
    }

    return params

  def get_best_tree(self):
    return self.estimator

  def get_tree_expression(self):
    return self.gpa.get_best_tree_expression()

  def score(self, x, y):
    return -1 * self.gpa.score(x, y)[0] # (-1 * self.gpa.score(x, y)[0], -1 * self.gpa.score(x, y)[1])