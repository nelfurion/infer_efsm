from pathlib import Path

from list_inputs_inference.base_estimator import BaseEstimator
from sklearn.model_selection import GridSearchCV

import operator
import math
import numbers


import pandas as pd

from custom_operators import protectedDivision, safe_binary_operation
from traces.trace_parser import TraceParser

global dir

class Estimator(BaseEstimator):
  def __init__(self, mu=10, lmbda=20, cxpb=0.2, mutpb=0.1, gcount=50, popsize=300, tournsize=None, tournparssize=None, selection='tournament'):
    self.set_params(mu, lmbda, cxpb, mutpb, gcount, popsize, selection, tournsize, tournparssize)
    self.inferrence_tree_file_name_prefix = dir

  def set_params(self, mu, lmbda, cxpb, mutpb, gcount, popsize, selection, tournsize=None, tournparssize=None):
    self.mu = mu
    self.lmbda = lmbda
    self.cxpb = cxpb
    self.mutpb = mutpb
    self.gcount = gcount
    self.popsize = popsize
    self.tournsize = tournsize
    self.selection = selection
    self.tournparssize = tournparssize

    self.setup = {
      'population_size': popsize,
      'hall_of_fame_size': 2,
      'input_list_length': 1, # hardcoding it to only accept a single argument # event_args_length,
      'output_type': float,
      'generations_count': gcount,
      'primitives': [
        # [safe_binary_operation(operator.add, 0), [float, float], float, 'add'],
        # [safe_binary_operation(operator.sub, 0), [float, float], float, 'sub'],
        # [safe_binary_operation(operator.mul, 0), [float, float], float, 'mul'],
        # [protectedDivision, [float, float], float, 'div']
        [operator.add, [float, float], float, 'add'],
        [operator.sub, [float, float], float, 'sub'],
        [operator.mul, [numbers.Complex, numbers.Complex], float, 'mul'],
        [operator.truediv, [numbers.Complex, numbers.Complex], float, 'div'],
      ],
      'terminals':[
        [1, float],
        [0, float]
      ],
      'individual_fitness_eval_func': self.eval_mean_squared_error,
      'selection': selection,
      'tournsize': tournsize,
      'tournparssize': tournparssize
    }

    self.estimator = None
    self.gpa = None

    return self

  def get_params(self, deep=False):
    params = {
      'mu': self.mu,
      'lmbda': self.lmbda,
      'cxpb': self.cxpb,
      'mutpb': self.mutpb,
      'gcount': self.gcount,
      'popsize': self.popsize,
      'selection': self.selection,
    }

    if self.tournsize:
      params['tournsize'] = self.tournsize

    if self.tournparssize:
      params['tournparssize'] = self.tournparssize

    return params

  def eval_mean_squared_error(self, individual, test_x_y_list=None, y_only_list=None):
        # Transform the tree expression in a callable function
        tree_expression = self.gpa.toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression
        # and the real function : x**4 + x**3 + x**2 + x

        # print('x_y_list: ', x_y_list)
        # print('self.target_list: ', self.target_list)

        squared_errors = []
        for x_y in (test_x_y_list or self.gpa.target_list):
          # print(func(x_y[0]))
          # print(x_y[1])
          try:
            # EDIT THIS
            # THIS IS JUST TEST IMPLEMENTATION OF RUNNING A FUNCTION MULTIPLE TIMES WITH A SINGLE PARAMETER

            # this is the coin event in the vending machine
            # the params for the coin event are from indexes 1 until the end of the array
            # FIX THIS PART
            params = x_y[0][1:]
            # print('params: ', params)

            registers = [0, 0, 0, 0, 0]
            output_condition_elements = []
            output_elements = []

            tree_expression_result = None
            # lets try to call the tree multiple times with a single parameter each time
            for param in params:

              # print('calling tree with param: ', param)
              # print('registers: ', registers)
              # pass the param in a list, so that we don't have to change the pick_array_element implementation
              # we will hardcode it to only work for the 0th index of the input, and also for 5 more indexes for 
              # custom registers.
              param_and_registers = [param] + registers # + [output_condition_elements, output_elements]
              # print('params list: ', params, ' CALLING WITH: ', param_and_registers)
              # print('TREE: ', individual)
              tree_expression_result = tree_expression(param_and_registers)
              registers = param_and_registers[-5:]
            # tree_expression_result = tree_expression(x_y[0]) // this is old code

            # only use the last tree expression result from above
            squared_error = (tree_expression_result - x_y[1]) ** 2
            squared_errors.append(squared_error)

            # print('------------------')
          except Exception as e: # if the tree is just: x , then we have array - integer
            return 100000,
        # squared_errors = () for x_y in self.target_list)

        return math.fsum(squared_errors) / len(squared_errors) if len(squared_errors) else 20000,

tp = TraceParser('./traces/vending_machine/traces_3855')
# tp = TraceParser('./traces/vending_machine/traces_9309')

event_args_length, events = tp.parse()



# TOURNAMENT SELECTION

x_y_list = events['coin']

# for i in range(21,22):
#   dir = './results/vending_machine/' + str(i) + '/'
#   Path(dir).mkdir(parents=True, exist_ok=True)
#   grid_search = GridSearchCV(
#     estimator=Estimator(dir), 
#     param_grid={
#       'mu': [5, 10],
#       'lmbda': [10], #[10, 20],
#       'cxpb': [0.1], #[0.1, 0.2],
#       'mutpb': [0.1], # [0.1, 0.2],
#       'gcount': [50, 1000],
#       'popsize': [100, 300, 1000],
#       'tournsize': [2, 4, 7],
#       'selection': ['tourn']
#     }
#   )

#   grid_search.fit(x_y_list, x_y_list)
#   dataframe = pd.DataFrame(grid_search.cv_results_)
#   dataframe.to_csv('./results/vending_machine/result' + '_' + str(i) + '.csv')


#  DCD TOURNAMENT SELECTION

# for i in range(23,24):
#   dir = './results/vending_machine/' + str(i) + '/'
#   Path(dir).mkdir(parents=True, exist_ok=True)
#   grid_search = GridSearchCV(
#     estimator=Estimator(dir), 
#     param_grid={
#       'mu': [5, 10],
#       'lmbda': [10], #[10, 20],
#       'cxpb': [0.1], #[0.1, 0.2],
#       'mutpb': [0.1], # [0.1, 0.2],
#       'gcount': [50],
#       'popsize': [100],
#       'selection': ['tourn_dcd']
#     }
#   )

#   grid_search.fit(x_y_list, x_y_list)
#   dataframe = pd.DataFrame(grid_search.cv_results_)
#   dataframe.to_csv('./results/vending_machine/result' + '_' + str(i) + '.csv')


#  DOUBLE TOURNAMENT SELECTION

for i in range(25,26):
  dir = './results/vending_machine/' + str(i) + '/'
  Path(dir).mkdir(parents=True, exist_ok=True)
  grid_search = GridSearchCV(
    estimator=Estimator(dir), 
    param_grid={
      'mu': [5, 10],
      'lmbda': [10], #[10, 20],
      'cxpb': [0.1], #[0.1, 0.2],
      'mutpb': [0.1], # [0.1, 0.2],
      'gcount': [50],
      'popsize': [100],
      'tournsize': [4],
      'tournparssize': [1.1, 1.4, 1.7],
      'selection': ['tourn_double']
    }
  )

  grid_search.fit(x_y_list, x_y_list)
  dataframe = pd.DataFrame(grid_search.cv_results_)
  dataframe.to_csv('./results/vending_machine/result' + '_' + str(i) + '.csv')