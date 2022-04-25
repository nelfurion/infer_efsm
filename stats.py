import re

from operator import attrgetter

from  deap.tools import MultiStatistics

class Stats(MultiStatistics):
  def __init__(self, **kwargs) -> None:
    self.records = []
    super().__init__(kwargs)

  def compile(self, population):
    res = super().compile(population)
    self.records.append(res)

    return res

  def compile_one(self, tree):
    res = super().compile([tree])

    return res

  # Get for the generation with the least fitness score.
  # Note: mulitple generations may have the same fitness score. This gets the first one.
  def get_best_generation_stats(self, best_tree):
    best_tree_string = str(best_tree)
    best_generation_stats = min(self.records, key=lambda r: r['fitness']['min'])
    best_generation_stats['gen_id'] = self.records.index(best_generation_stats)
    best_generation_stats['best_tree_size'] = self.compile_one(best_tree)['size']['min']
    best_generation_stats['reg_set_count'] = best_tree_string.count("set_")
    best_generation_stats['reg_read_count'] = len(re.findall(r'pick_\w+_\d', best_tree_string))
    best_generation_stats['reg_count'] = len(set(re.findall(r'set_\d', best_tree_string)))

    return best_generation_stats
    
  def get_best_generation_stats_string(self, best_tree):
    stats = self.get_best_generation_stats(best_tree)

    string = ','.join([
      str(stats['gen_id']),
      str(stats['best_tree_size']),
      str(stats['reg_set_count']),
      str(stats['reg_read_count']),
      str(stats['reg_count'])
    ])


    return string