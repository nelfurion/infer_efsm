import operator

import math, numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from custom_operators import generate_index_combinations, pick_arr_el, protectedDivision, safe_binary_operation, subtract_list_elements, sum_list_elements
from plot import plot_tree, plot_two, plot_two_2
from traces.trace_parser import TraceParser

class GPListInputAlgorithm:
    def __init__(self, population_size, hof_size, input_types, output_type, list_length, generations_count) -> None:
        self.pset = gp.PrimitiveSetTyped("MAIN", input_types, output_type)
        self.toolbox = base.Toolbox()
        self.population_size = population_size
        self.hof = tools.HallOfFame(hof_size)

        self.list_length = list_length
        self.generations_count = generations_count

        self.pset.renameArguments(ARG0="x")
        self.pset.renameArguments(ARG1="y")
        self.pset.renameArguments(ARG2="z")
        self.pset.renameArguments(ARG3="q")
        self.pset.renameArguments(ARG4="r")
        self.pset.renameArguments(ARG5="s")

        self.add_stats()
        self.addFitness((-1.0,))
        self.addIndividual(gp.PrimitiveTree)
        self.addNecessaryPrimitives()

    @staticmethod
    def create(setup):
      gpa =  GPListInputAlgorithm(
        population_size=setup['population_size'],
        hof_size=setup['hall_of_fame_size'],
        input_types=setup['input_types'] if 'input_types' in setup.keys() else [list],
        output_type=setup['output_type'],
        list_length=setup['input_list_length'],
        generations_count=setup['generations_count']
      )

      gpa.addPrimitives(setup['primitives'])
      gpa.addTerminals(setup['terminals'])
      gpa.set_target(setup['target'])

      return gpa

    def addNecessaryPrimitives(self):
      """
        These primitives are necessary to make the algorithm run.
        They are separate from what you would add in order to infer a function.
        
        pick_arr_el - picks a specific element from the input array so that it can be used in normal primitives - like operator.mul
                    A separate pick_arr_el is generated for each index of the input list, so that the GP can decide which array element to use.
        
        lambda x: x - used for float and list is the minimal implementation DEAP needs in order to be able to extend trees.
      """

      # Adds the primitive for each separate index of the input list, so that we can choose which argument to use at each step.
      for i in range(self.list_length):
        self.addPrimitive(pick_arr_el(i), [list], float, 'pick_' + str(i))

      # Identity primitives - the minimal implementation necessary to allow extending trees.
      self.addPrimitive(lambda x: x, [list], list, 'list_list')
      self.addPrimitive(lambda x: x, [float], float, 'float_float')

    def addPrimitive(self, operator, input_types, output_type, name):
        self.pset.addPrimitive(operator, input_types, output_type, name)

    def addPrimitives(self, primitives):
      for primitive_args in primitives:
        self.addPrimitive(*primitive_args)

    def addTerminal(self, value, type):
        self.pset.addTerminal(value, type)

    def addTerminals(self, terminals):
      for terminal_args in terminals:
        self.addTerminal(*terminal_args)

    def addEphemeralConstant(self, name, func):
        self.pset.addEphemeralConstant(name, func)        

    def addFitness(self, weights):
        creator.create("FitnessMin", base.Fitness, weights=weights)

    def addIndividual(self, individualClass):
        creator.create("Individual", individualClass, fitness=creator.FitnessMin)

    def set_target(self, target_list) -> None:
        self.target_list = target_list

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
        # print('______eval_mean_squared_error')
        # print(individual)
        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression
        # and the real function : x**4 + x**3 + x**2 + x

        squared_errors = []
        for x_y in self.target_list:
          # print(func(x_y[0]))
          # print(x_y[1])
          try:
            # print('TREE INPUTS: ', x_y[0])
            # print('arg - res: ', x_y[0], ' - ', func(x_y[0]))
            squared_error = (func(x_y[0]) - x_y[1]) ** 2
            squared_errors.append(squared_error)
          except TypeError: # if the tree is just: x , then we have array - integer
            print('error')
            pass
        # squared_errors = () for x_y in self.target_list)

        return math.fsum(squared_errors) / len(squared_errors) if len(squared_errors) else 20000,
    
    def add_stats(self):
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self.mstats.register("avg", numpy.mean)
        self.mstats.register("std", numpy.std)
        self.mstats.register("min", numpy.min)
        self.mstats.register("max", numpy.max)

    def run(self):
      self.addTools()
      population = self.toolbox.population(n=self.population_size)
      population, log = algorithms.eaMuCommaLambda(population, self.toolbox, 10, 20, 0.2, 0.1, self.generations_count, stats=self.mstats,
                                  halloffame=self.hof, verbose=True)

    def get_best_tree(self):
        return self.hof[0]

    def get_best_tree_expression(self):
        return self.toolbox.compile(expr=self.get_best_tree())