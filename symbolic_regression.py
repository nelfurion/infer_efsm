import operator

import math, random, numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from plot import plot_tree, plot_two

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))

pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

POINTS = [x/10. for x in range(-10,10)]
def target_regression(x):
  return x**4 - x**3 - x**2 - x

def evalSymbolicRegression(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    squared_errors = ((func(x) - target_regression(x))**2 for x in points)
    return math.fsum(squared_errors) / len(points),

toolbox.register("evaluate", evalSymbolicRegression, points=POINTS)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Then, we decorate the mate and mutate method to limit the height of generated individuals. 
# This is done to avoid an important draw back of genetic programming : bloat. 
# Koza in his book on genetic programming suggest to use a max depth of 17.
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)

population = toolbox.population(n=300)
hof = tools.HallOfFame(1)
# algorithms.eaSimple() - frequently the best score is around 0.02 etc

population, log = algorithms.eaMuCommaLambda(population, toolbox, 10, 20, 0.2, 0.1, 340, stats=mstats,
                                   halloffame=hof, verbose=True)

best_tree = hof[0]
plot_tree(best_tree)
best_tree_score = evalSymbolicRegression(best_tree, POINTS)
print("Best Tree Syntax: ", str(best_tree))
print("Best Tree Score: ", best_tree_score)

best_tree_expression = toolbox.compile(expr=best_tree)
plot_two(POINTS, target_regression, best_tree_expression)