from deap import algorithms

from gp_algorithm import GPListInputAlgorithm
from lib import generate_random_string
from plot import plot_tree

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
    try:
      plot_tree(self.estimator, self.inferrence_tree_file_name_prefix + str(self.gpa.score(target_x_y, y)[0]) + "--" + str(self.get_params()) + generate_random_string(10))
    except Exception as e:
      print(e)

    return self

  def get_best_tree(self):
    return self.estimator

  def get_tree_expression(self):
    return self.gpa.get_best_tree_expression()

  def score(self, x, y):
    return -1 * self.gpa.score(x, y)[0]