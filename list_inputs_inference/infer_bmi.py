from list_inputs_inference.base_estimator import BaseEstimator
from traces.trace_parser import TraceParser

import math

class Estimator(BaseEstimator):
  def __init__(self, mu=None, lmbda=None, cxpb=None, mutpb=None, gcount=None, popsize=None, mut_tool=None, cx_tool=None, selection=None, tree_output_dir=None, tournsize=None, tournparssize=None):
    self.set_params(mu, lmbda, cxpb, mutpb, gcount, popsize, mut_tool, cx_tool, selection, tree_output_dir, tournsize, tournparssize)

  def get_params(self, deep=False):
    return {
      'mu': self.mu,
      'lmbda': self.lmbda,
      'cxpb': self.cxpb,
      'mutpb': self.mutpb,
    }

  # # test_x_y_list is None during training. During training the self.gpa.target_list is used.
  # # When testing the score of the already trained tree - test_x_y_list is used.
  # # We split the data into training and testing sets.
  # def mean_squared_error_bmi(self, individual, test_x_y_list = None, y_only_list = None):
  #   tree_expression = self.gpa.toolbox.compile(expr=individual)
  #   squared_errors = []

  #   for x, y in (test_x_y_list or self.gpa.target_list):
  #     params = x[0:-1] # here particularly the last param is an empty string which we shouldn't use
  #     registers = [0, 0, 0, 0, 0]
  #     res = tree_expression(params + registers)
  #     squared_error = (res - float(y)) ** 2

  #     squared_errors.append(squared_error)
    
  #   return math.fsum(squared_errors) / len(squared_errors),

  # Fitness objective is to minimize the number of errors
  def fitness_eval_fun(self, individual, test_x_y_list = None, y_only_list = None):
    tree_expression = self.gpa.toolbox.compile(expr=individual)
    squared_errors = []
    # x is a list with a single param [param]
    # y is a value
    for x, y in (test_x_y_list or self.gpa.target_list):
      registers = [None, None, None, None, None]

      all_inputs = x + registers
      try:
        res = tree_expression(all_inputs)
        if (res != y):
          squared_errors.append(1)
      except (TypeError, ValueError, ZeroDivisionError) as e:
        return math.inf,


    # if_discount = min(str(individual).count('if_else') * 10, 10)

    # gt_discount = min(str(individual).count('gt') * 10, 10)
    # lt_discount = min(str(individual).count('lt') * 10, 10)
    # num_1_discount = min(str(individual).count('18.5') * 10, 10)
    # num_2_discount = min(str(individual).count('24.9') * 10, 10)
    # pick_1_discount = min(str(individual).count('pick_float_0') * 10, 10)
    # pick_2_discount = min(str(individual).count('pick_float_1') * 10, 10)

    return len(squared_errors),# - if_discount - gt_discount - lt_discount - num_1_discount - num_2_discount - pick_1_discount - pick_2_discount,
    


tp = TraceParser('./traces/bmi/traces_9728')

event_args_length, events = tp.parse()

x_y_list = events['bmi']
y_list = list(map(lambda s: s[-1], x_y_list))

print(x_y_list[0:5])
print(y_list[0:5])

# print(events['bmi'])

# train_set = events['bmi'][:4700]
# test_set = events['bmi'][4700:]


# gpa_estimator = Estimator()
# gpa_estimator.fit(train_set, train_set)


# print('**************************')

# best_tree_score = gpa_estimator.score(test_set, test_set)
# print("Best Tree Syntax: ", str(gpa_estimator.get_best_tree()))
# print("Best Tree Score on test set: ", best_tree_score)
# plot_tree(gpa_estimator.get_best_tree())
# print('HOF 2nd best')
# print(gpa_estimator.gpa.hof[1])

# best_tree_exp = gpa_estimator.get_tree_expression()
# print('best tree outputs 0:10')
# for x, y in test_set[0:10]:
#   print(x, best_tree_exp(x + [None, None, None, None, None, [], []]))



# x_y_list = events['bmi']

# grid_search_tree = GridSearchCV(
#   estimator=Estimator(), 
#   param_grid={
#     'mu': [5, 10],
#     'lmbda': [10], #[10, 20],
#     'cxpb': [0.1], #[0.1, 0.2],
#     'mutpb': [0.1], # [0.1, 0.2],
#   }
# )

# grid_search_tree.fit(x_y_list, x_y_list)
# dataframe = pd.DataFrame(grid_search_tree.cv_results_)
# dataframe.to_csv('result.csv')