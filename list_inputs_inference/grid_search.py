from sklearn.model_selection import GridSearchCV

import operator

import pandas as pd

from deap import algorithms

from custom_operators import protectedDivision, safe_binary_operation
from plot import plot_tree, plot_two_2, plot_3d
from traces.trace_parser import TraceParser

from gp_algorithm import GPListInputAlgorithm

class Estimator:
  def __init__(self, mu=10, lmbda=20, cxpb=0.2, mutpb=0.1):
    self.set_params(mu, lmbda, cxpb, mutpb)

  def set_params(self, mu, lmbda, cxpb, mutpb):
    self.mu = mu
    self.lmbda = lmbda
    self.cxpb = cxpb
    self.mutpb = mutpb

    self.setup = {
      'population_size': 300,
      'hall_of_fame_size': 2,
      'input_list_length': 1, # hardcoding it to only accept a single argument # event_args_length,
      'output_type': float,
      'generations_count': 1000,
      'primitives': [
        [safe_binary_operation(operator.add, 0), [float, float], float, 'add'],
        [safe_binary_operation(operator.sub, 0), [float, float], float, 'sub'],
        [safe_binary_operation(operator.mul, 0), [float, float], float, 'mul'],
        [protectedDivision, [float, float], float, 'div']
      ],
      'terminals':[
        [0, float]
      ],
    }

    self.estimator = None
    self.gpa = None

    return self

  def get_params(self, deep=False):
    return {
      'mu': self.mu,
      'lmbda': self.lmbda,
      'cxpb': self.cxpb,
      'mutpb': self.mutpb,
    }

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

    return self

  def get_best_tree(self):
    return self.estimator

  def get_tree_expression(self):
    return self.gpa.get_best_tree_expression()

  def score(self, x, y):
    return -1 * self.gpa.score(x, y)[0]


tp = TraceParser('./traces/vending_machine/traces_3855')

event_args_length, events = tp.parse()

# print(events['coin'])

# train_set = events['coin'][:300]
# test_set = events['coin'][300:]


# gpa_estimator = Estimator()
# gpa_estimator.fit(train_set, train_set)


# best_tree_score = gpa_estimator.score(test_set, test_set)
# print("Best Tree Syntax: ", str(gpa_estimator.get_best_tree()))
# print("Best Tree Score: ", best_tree_score)
# plot_tree(gpa_estimator.get_best_tree())


x_y_list = events['coin']

grid_search_tree = GridSearchCV(
  estimator=Estimator(), 
  param_grid={
    'mu': [5, 10],
    'lmbda': [10, 20],
    'cxpb': [0.1, 0.2],
    'mutpb': [0.1, 0.2],
  }
)

grid_search_tree.fit(x_y_list, x_y_list)
dataframe = pd.DataFrame(grid_search_tree.cv_results_)
dataframe.to_csv('result.csv')