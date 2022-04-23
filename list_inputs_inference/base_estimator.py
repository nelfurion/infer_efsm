from deap import algorithms

from gp_algorithm import GPListInputAlgorithm
from lib import generate_random_string
from plot import plot_tree
import uuid
import pandas as pd

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
    logbook = self.gpa.run()   
    self.estimator = self.gpa.get_best_tree()

    # try:
    #   id = str(uuid.uuid4())
    #   filepath = self.tree_output_dir + str(self.gpa.score(target_x_y, y)[0]) + "--" + id
    #   plot_tree(self.estimator, filepath)
      
    #   logbook_dataframe = pd.DataFrame(logbook)
    #   logbook_dataframe.to_csv(filepath + '--log.csv', index=False)

    #   with open(filepath + '--params.txt', 'w') as the_file:
    #     the_file.write(str(self.get_params()))

    #   print('Tree saved to file: ' + filepath)
    # except Exception as e:
    #   print(e)

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
      'tournparssize': self.tournparssize or 'N/A'
    }

    return params

  def get_best_tree(self):
    return self.estimator

  def get_tree_expression(self):
    return self.gpa.get_best_tree_expression()

  def score(self, x, y):
    return -1 * self.gpa.score(x, y)[0]