from deap import gp
from numpy import *
import pygraphviz as pgv

import math
import matplotlib.pyplot as plt

def plot_tree(tree):
  nodes, edges, labels = gp.graph(tree)

  g = pgv.AGraph()
  g.add_nodes_from(nodes)
  g.add_edges_from(edges)
  g.layout(prog="dot")

  for i in nodes:
      n = g.get_node(i)
      n.attr["label"] = labels[i]

  g.draw("tree.pdf")

def plot_two(x_list, base_f, f2):
  y1_list = [base_f(x) for x in x_list]
  y2_list = [f2(x) for x in x_list]
  plt.plot(x_list, y1_list, 'g', label='BASE')
  plt.plot(x_list, y2_list, 'r', label='CANDIDATE')
  plt.legend()
  plt.show()