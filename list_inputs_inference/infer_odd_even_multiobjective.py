from sklearn.model_selection import GridSearchCV

from plot import plot_tree
from traces.trace_parser import TraceParser

from list_inputs_inference.base_estimator import BaseEstimator

import math


class Estimator(BaseEstimator):
  def __init__(
    self,
    mu=None,
    lmbda=None,
    cxpb=None,
    mutpb=None,
    gcount=None,
    popsize=None,
    mut_tool=None,
    cx_tool=None,
    selection=None,
    tree_output_dir=None,
    tournsize=None,
    tournparssize=None,
    fitness_weights=None,
    output_type=str
  ):
    self.set_params(
      mu,
      lmbda,
      cxpb,
      mutpb,
      gcount,
      popsize,
      mut_tool,
      cx_tool,
      selection,
      tree_output_dir,
      tournsize,
      tournparssize,
      fitness_weights,
      output_type
    )

  def appearance_fitness_eval(self, individual):
    return 1

  # test_x_y_list is None during training. During training the self.gpa.target_list is used.
  # When testing the score of the already trained tree - test_x_y_list is used.
  # We split the data into training and testing sets.
  #
  # Returns
  #   (actual value, reallistic max value)
  def expression_fitness_eval(self, individual, test_x_y_list = None, y_only_list = None):
    tree_expression = self.gpa.toolbox.compile(expr=individual)
    squared_errors = []
    # x is a list with a single param [param]
    # y is a value
    for x, y in (test_x_y_list or self.gpa.target_list):
      registers = [None, None, None, None, None]
      try:
        res = tree_expression(x + registers)
        if (res != y):
          squared_errors.append(1)
      except (TypeError, ValueError, ZeroDivisionError) as e:
        return math.inf, math.inf
  
    return len(squared_errors), len(test_x_y_list or self.gpa.target_list)

  def fitness_eval_fun(self, individual, test_x_y_list = None, y_only_list = None):
    expr_error_score, expr_max_error_score = self.expression_fitness_eval(individual, test_x_y_list)
    if (expr_error_score == math.inf):
      return math.inf,

    appearance_error_score = self.appearance_fitness_eval(individual)

    expr_ratio = expr_error_score / expr_max_error_score

    return expr_error_score,



tp = TraceParser('./traces/is_even/traces_6991')


event_args_length, events = tp.parse()

x_y_list = events['is_even']
y_list = list(map(lambda s: s[-1], x_y_list))

print(x_y_list[0:10])
print(y_list[0:10])

# event_args_length, events = tp.parse()

# print(events['is_even'])

# train_set = events['is_even'][:4700]
# test_set = events['is_even'][300:]


# gpa_estimator = Estimator()
# gpa_estimator.fit(train_set, train_set)


# print('**************************')

# best_tree_score = gpa_estimator.score(test_set, test_set)
# print("Best Tree Syntax: ", str(gpa_estimator.get_best_tree()))
# print("Best Tree Score: ", best_tree_score)
# plot_tree(gpa_estimator.get_best_tree())
# best_tree_exp = gpa_estimator.get_tree_expression()
# print('best tree outputs 0:10')
# for x, y in test_set[0:10]:
#   print(x, best_tree_exp(x + [None, None, None, None, None, [], []]))

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