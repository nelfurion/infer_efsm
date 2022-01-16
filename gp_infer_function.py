import operator

import math, random, numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from custom_operators import protectedDivision
from plot import plot_tree, plot_two

class GPAlgorithm:
    def __init__(self, population_size, hof_size) -> None:
        self.pset = gp.PrimitiveSet("MAIN", 1)
        self.toolbox = base.Toolbox()
        self.population_size = population_size
        self.hof = tools.HallOfFame(hof_size)

        self.pset.renameArguments(ARG0="x")
        self.pset.renameArguments(ARG1="y")
        self.pset.renameArguments(ARG2="z")
        self.pset.renameArguments(ARG3="q")
        self.pset.renameArguments(ARG4="r")
        self.pset.renameArguments(ARG5="s")

        self.add_stats()
        self.addFitness((-1.0,))
        self.addIndividual(gp.PrimitiveTree)

    def addPrimitive(self, operator, arg_count):
        self.pset.addPrimitive(operator, arg_count)

    def addEphemeralConstant(self, name, func):
        self.pset.addEphemeralConstant(name, func)        

    def addFitness(self, weights):
        creator.create("FitnessMin", base.Fitness, weights=weights)

    def addIndividual(self, individualClass):
        creator.create("Individual", individualClass, fitness=creator.FitnessMin)

    def set_target(self, target_hash) -> None:
        self.target_hash = target_hash

    def addTools(self):
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.toolbox.register("evaluate", self.eval_mean_squared_error)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

        # Then, we decorate the mate and mutate method to limit the height of generated individuals. 
        # This is done to avoid an important draw back of genetic programming : bloat. 
        # Koza in his book on genetic programming suggest to use a max depth of 17.
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    def eval_mean_squared_error(self, individual):
        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression
        # and the real function : x**4 + x**3 + x**2 + x
        squared_errors = ((func(x) - self.target_hash[x])**2 for x in self.target_hash.keys())

        return math.fsum(squared_errors) / len(self.target_hash.keys()),
    
    def add_stats(self):
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self.mstats.register("avg", numpy.mean)
        self.mstats.register("std", numpy.std)
        self.mstats.register("min", numpy.min)
        self.mstats.register("max", numpy.max)

    def run(self):
        population = self.toolbox.population(n=300)
        population, log = algorithms.eaMuCommaLambda(population, self.toolbox, 10, 20, 0.2, 0.1, 340, stats=self.mstats,
                                   halloffame=self.hof, verbose=True)

    def get_best_tree(self):
        return self.hof[0]

    def get_best_tree_expression(self):
        return self.toolbox.compile(expr=best_tree)


# pset = gp.PrimitiveSet("MAIN", 1)
# pset.addPrimitive(operator.add, 2)
# pset.addPrimitive(operator.sub, 2)
# pset.addPrimitive(operator.mul, 2)
# pset.addPrimitive(protectedDivision, 2)
# pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(math.cos, 1)
# pset.addPrimitive(math.sin, 1)
# pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))

# pset.renameArguments(ARG0="x")
# pset.renameArguments(ARG1="y")

# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# toolbox = base.Toolbox()
# toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
# toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox.register("compile", gp.compile, pset=pset)

# POINTS = [x/10. for x in range(-10,10)]
# def target_regression(x):
#   return x**4 - x**3 - x**2 - x

# def evalSymbolicRegression(individual, points):
#     # Transform the tree expression in a callable function
#     func = toolbox.compile(expr=individual)
#     # Evaluate the mean squared error between the expression
#     # and the real function : x**4 + x**3 + x**2 + x
#     squared_errors = ((func(x) - target_regression(x))**2 for x in points)
#     return math.fsum(squared_errors) / len(points),

# toolbox.register("evaluate", evalSymbolicRegression, points=POINTS)
# toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("mate", gp.cxOnePoint)
# toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
# toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# # Then, we decorate the mate and mutate method to limit the height of generated individuals. 
# # This is done to avoid an important draw back of genetic programming : bloat. 
# # Koza in his book on genetic programming suggest to use a max depth of 17.
# toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
# toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


# stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
# stats_size = tools.Statistics(len)
# mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
# mstats.register("avg", numpy.mean)
# mstats.register("std", numpy.std)
# mstats.register("min", numpy.min)
# mstats.register("max", numpy.max)

# population = toolbox.population(n=300)
# hof = tools.HallOfFame(1)
# algorithms.eaSimple() - frequently the best score is around 0.02 etc



# best_tree = hof[0]

target = {
    1: 1,
    2: 13,
    3: 24,
    4: 1,
    5: 34,
    6: 12
}

gpa = GPAlgorithm(300, 1)
gpa.addPrimitive(operator.add, 2)
gpa.addPrimitive(operator.sub, 2)
gpa.addPrimitive(operator.mul, 2)
gpa.addPrimitive(protectedDivision, 2)
gpa.addPrimitive(operator.neg, 1)
gpa.addPrimitive(math.cos, 1)
gpa.addPrimitive(math.sin, 1)
gpa.addEphemeralConstant("rand101", lambda: random.randint(-1,1))

gpa.set_target(target)
gpa.addTools()
gpa.run()

best_tree = gpa.get_best_tree()
plot_tree(best_tree)
best_tree_score = gpa.eval_mean_squared_error(best_tree)
print("Best Tree Syntax: ", str(best_tree))
print("Best Tree Score: ", best_tree_score)

best_tree_expression = gpa.get_best_tree_expression()
target_func = lambda key: target[key]
plot_two(target.keys(), target_func, best_tree_expression)