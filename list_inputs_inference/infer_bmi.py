from sklearn.model_selection import GridSearchCV

import operator, math, numbers

import pandas as pd

from deap import algorithms

from custom_operators import protectedDivision, safe_binary_operation
from plot import plot_tree, plot_two_2, plot_3d
from traces.trace_parser import TraceParser

from gp_algorithm import GPListInputAlgorithm



class Estimator:
  def __init__(self, mu=10, lmbda=20, cxpb=0.8, mutpb=0.2):
    self.set_params(mu, lmbda, cxpb, mutpb)

  def set_params(self, mu, lmbda, cxpb, mutpb):
    self.mu = mu
    self.lmbda = lmbda
    self.cxpb = cxpb
    self.mutpb = mutpb

    self.setup = {
      'population_size': 1000,
      'hall_of_fame_size': 2,
      'input_list_length': 1, # hardcoding it to only accept a single argument # event_args_length,
      'output_type': str,
      'generations_count': 10000,
      'primitives': [
        # [operator.add, [float, float], float, 'add'],
        # [operator.sub, [float, float], float, 'sub'],
        [operator.mul, [numbers.Complex, numbers.Complex], float, 'mul'],
        [operator.truediv, [numbers.Complex, numbers.Complex], float, 'div'],
        # [safe_binary_operation(operator.mod, -1), [float, float], float, 'mod'],
        # [operator.ge, [float, float], bool, 'ge'],
        # [operator.gt, [float, float], bool, 'gt'],
        # [operator.le, [float, float], bool, 'le'],
        [operator.lt, [numbers.Complex, numbers.Complex], bool, 'lt'],
        
        # [operator.eq, [float, float], bool, 'eq'],
        # [operator.not_, [bool], bool, 'my_not'],
        [lambda bool_1, bool_2: bool_1 and bool_2, [bool, bool], bool, 'my_and'],
        # [lambda bool_1, bool_2: bool_1 or bool_2, [bool, bool], bool, 'my_or'],
        # [lambda bool_1, bool_2: bool_1 != bool_2, [bool, bool], bool, 'my_xor'],
        # [lambda string_1, string_2, boolean: string_1 if boolean else string_2, [str, str, bool], str, 'str_on_b'],
        # [lambda is_even: 'yes' if is_even else 'no', [bool], str, 'is_even'],
        # [lambda is_overweight: 'overweight' if is_overweight else 'healthy', [bool], str, 'is_overweight'],
        # [lambda is_overweight: 'overweight' if is_overweight else 'healthy', [bool], str, 'is_underweight']
        # [lambda is_even: 'yes' if is_even else 'no', [bool], str, 'is_even']
        # [lambda is_even: 'yes' if is_even else 'no', [bool], str, 'is_even']
      ],
      'terminals':[
        # [0, float],
        # [1, float],
        # [2, float],
        [10000, float],
        # [100, float],
        [True, bool],
        [False, bool],
        ['healthy', str],
        ['underweight', str],
        ['overweight', str]
      ],
      'individual_fitness_eval_func': self.score_same_classes
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
    return self.gpa.score(x, y)[0]

  # test_x_y_list is None during training. During training the self.gpa.target_list is used.
  # When testing the score of the already trained tree - test_x_y_list is used.
  # We split the data into training and testing sets.
  def mean_squared_error_bmi(self, individual, test_x_y_list = None, y_only_list = None):
    tree_expression = self.gpa.toolbox.compile(expr=individual)
    squared_errors = []

    for x, y in (test_x_y_list or self.gpa.target_list):
      params = x[0:-1] # here particularly the last param is an empty string which we shouldn't use
      registers = [0, 0, 0, 0, 0]
      res = tree_expression(params + registers)
      squared_error = (res - float(y)) ** 2

      squared_errors.append(squared_error)
    
    return math.fsum(squared_errors) / len(squared_errors),

  # Fitness objective is to minimize the number of errors
  def score_same_classes(self, individual, test_x_y_list = None, y_only_list = None):
    tree_expression = self.gpa.toolbox.compile(expr=individual)
    squared_errors = []
    # x is a list with a single param [param]
    # y is a value
    for x, y in (test_x_y_list or self.gpa.target_list):
      registers = [None, None, None, None, None]
      output_condition_elements = []
      output_elements = []

      # [1, 2, 0, 0, 0, 0, 0, [], []]
      all_inputs = x + registers + [output_condition_elements, output_elements]
      try:
        res = tree_expression(all_inputs)
        if (res != y):
          squared_errors.append(1)
      except (TypeError, ValueError, ZeroDivisionError) as e:
        return 100000,

    return len(squared_errors),
    


tp = TraceParser('./traces/bmi/traces_9728')

event_args_length, events = tp.parse()

# print(events['bmi'])

train_set = events['bmi'][:4700]
test_set = events['bmi'][4700:]


gpa_estimator = Estimator()
gpa_estimator.fit(train_set, train_set)


print('**************************')

best_tree_score = gpa_estimator.score(test_set, test_set)
print("Best Tree Syntax: ", str(gpa_estimator.get_best_tree()))
print("Best Tree Score on test set: ", best_tree_score)
plot_tree(gpa_estimator.get_best_tree())
print('HOF 2nd best')
print(gpa_estimator.gpa.hof[1])

best_tree_exp = gpa_estimator.get_tree_expression()
print('best tree outputs 0:10')
for x, y in test_set[0:10]:
  print(x, best_tree_exp(x + [None, None, None, None, None, [], []]))



# x_y_list = events['is_even']

# grid_search_tree = GridSearchCV(
#   estimator=Estimator(), 
#   param_grid={
#     'mu': [5, 10],
#     'lmbda': [10, 20],
#     'cxpb': [0.1, 0.2],
#     'mutpb': [0.1, 0.2],
#   }
# )

# grid_search_tree.fit(x_y_list, x_y_list)
# dataframe = pd.DataFrame(grid_search_tree.cv_results_)
# dataframe.to_csv('result.csv')