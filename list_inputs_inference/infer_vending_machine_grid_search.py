import copy
from pathlib import Path

from list_inputs_inference.base_estimator import BaseEstimator
from sklearn.model_selection import GridSearchCV

import math

import pandas as pd

from custom_operators import protectedDivision, safe_binary_operation
from traces.trace_parser import TraceParser

class Estimator(BaseEstimator):
  def __init__(self, mu=None, lmbda=None, cxpb=None, mutpb=None, gcount=None, popsize=None, mut_tool=None, cx_tool=None, selection=None, tree_output_dir=None, tournsize=None, tournparssize=None):
    self.set_params(mu, lmbda, cxpb, mutpb, gcount, popsize, mut_tool, cx_tool, selection, tree_output_dir, tournsize, tournparssize)

  # MEAN SQUARED ERORR ON LOOP
  def fitness_eval_fun(self, individual, test_x_y_list=None, y_only_list=None):
        # Transform the tree expression in a callable function
        tree_expression = self.gpa.toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression
        # and the real function : x**4 + x**3 + x**2 + x

        squared_errors = []
        for x_y in (test_x_y_list or self.gpa.target_list):
          try:
            # EDIT THIS
            # THIS IS JUST TEST IMPLEMENTATION OF RUNNING A FUNCTION MULTIPLE TIMES WITH A SINGLE PARAMETER

            # this is the coin event in the vending machine
            # the params for the coin event are from indexes 1 until the end of the array
            # FIX THIS PART
            params = x_y[0][1:]

            registers = [0, 0, 0, 0, 0]
            tree_expression_result = None

            # lets try to call the tree multiple times with a single parameter each time
            for param in params:
              # pass the param in a list, so that we don't have to change the pick_array_element implementation
              # we will hardcode it to only work for the 0th index of the input, and also for 5 more indexes for 
              # custom registers.
              param_and_registers = [param] + registers # + [output_condition_elements, output_elements]

              tree_expression_result = tree_expression(param_and_registers)
              registers = param_and_registers[-5:]

            # only use the last tree expression result from above
            squared_error = (tree_expression_result - x_y[1]) ** 2
            squared_errors.append(squared_error)

          except Exception as e: # if the tree is just: x , then we have array - integer
            # import traceback
            # print(e)
            # print(traceback.format_exc())
            return math.inf,

        return math.sqrt(math.fsum(squared_errors) / len(squared_errors)) if len(squared_errors) else math.inf,

# tp = TraceParser('./traces/vending_machine/traces_3855')
# tp = TraceParser('./traces/vending_machine/traces_9309')
tp = TraceParser('./traces/vending_machine/traces_1703')

event_args_length, events = tp.parse()

x_y_list = events['coin']
y_list = list(map(lambda s: s[-1], x_y_list))
