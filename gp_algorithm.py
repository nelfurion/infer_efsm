import operator

import numpy
import math

from mpmath import exp

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from stats import Stats

# from scoop import futures

from custom_operators import pick_arr_el, set_arr_el

REGISTERS_COUNT = 5

# Each tree receives an input list
# First n elements of the list are the actual function inputs
# Then next REGISTERS_COUNT elements are 0s - empty register values
# The last element in the list is a list. 
#   The tree would use this list to fill it with all registers which value needs to be checked to produce an output.
#   The idea is that we may have N number of if statements - each of which if true produces a different output.
#   E.g. the BMI function:
#   The possible outcomes are 3 different strings, and the tree needs 3 if conditions to decide which string to output.
#   This is a way to make the tree more generic - e.g. without the need to add an if-else-if-then operator, as if we add more 
#   possible outputs to that function we may need an if-elseif-elseif-elseif-then operator and so on.

class GPListInputAlgorithm:
    def __init__(self, population_size, hof_size, input_types, output_type, list_length, generations_count, individual_fitness_eval_func, mut_tool, cx_tool, selection, tournsize=None, tournparssize=None) -> None:
        self.pset = gp.PrimitiveSetTyped("MAIN", input_types, output_type)
        self.output_type = output_type
        self.toolbox = base.Toolbox()
        self.population_size = population_size
        self.hof = tools.HallOfFame(hof_size)

        self.list_length = list_length
        self.generations_count = generations_count

        self.individual_fitness_eval_func = individual_fitness_eval_func

        self.mut_tool = mut_tool
        self.cx_tool = cx_tool
        self.selection = selection
        self.tournsize = tournsize
        self.tournparssize = tournparssize

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

        # for when selecting semantic mucation or semantic crossover both of which require the addition of the lf function
        self.lf_func_added = False

    @staticmethod
    def create(setup):
      gpa =  GPListInputAlgorithm(
        population_size=setup['population_size'],
        hof_size=setup['hall_of_fame_size'],
        input_types=setup['input_types'] if 'input_types' in setup.keys() else [list],
        output_type=setup['output_type'],
        list_length=setup['input_list_length'],
        generations_count=setup['generations_count'],
        individual_fitness_eval_func=setup['individual_fitness_eval_func'] if 'individual_fitness_eval_func' in setup.keys() else None,
        mut_tool=setup['mut_tool'],
        cx_tool=setup['cx_tool'],
        selection=setup['selection'],
        tournsize=setup['tournsize'],
        tournparssize=setup['tournparssize']
      )

      gpa.setPopulationGenerationFunc(setup['population_generation_func'])
      gpa.addPrimitives(setup['primitives'])
      gpa.addTerminals(setup['terminals'])
      gpa.set_target(setup['target'])

      gpa.addTools()

      return gpa

    def setPopulationGenerationFunc(self, population_gen_func):
        self.population_gen_func = population_gen_func

    def addNecessaryPrimitives(self):
      """
        These primitives are necessary to make the algorithm run.
        They are separate from what you would add in order to infer a function.
        
        pick_arr_el - picks a specific element from the input array so that it can be used in normal primitives - like operator.mul
                    A separate pick_arr_el is generated for each index of the input list, so that the GP can decide which array element to use.
        
        lambda x: x - used for float and list is the minimal implementation DEAP needs in order to be able to extend trees.
      """

      self.addDataStoreAndRetrievePrimitives()

      # Identity primitives - the minimal implementation necessary to allow extending trees.
      self.addPrimitive(lambda x: x, [list], list, 'list_list')
      self.addPrimitive(lambda x: x, [float], float, 'float_float')

    def addDataStoreAndRetrievePrimitives(self):
      # The - 2 in both pick/set loops is for the last input elements which would be lists used internally to store boolean 
      # statements that need to be checked to produce different outputs, and the indexes of elements to use as outputs.

      # Adds the primitive for each separate index of the input list, so that we can choose which argument to use at each step.
      # pick_3(x) means take the x input which is the array of [input1_, input_2, ..., reg_1, reg_2, ...] and take the 3rd element
      # The 3rd element can either be an input value or a register value.
      for i in range(self.list_length + REGISTERS_COUNT): # add + 5 for 5 additional registers which are initially set to 0
        self.addPrimitive(pick_arr_el(i, float), [list], float, 'pick_float_' + str(i))
        # self.addPrimitive(pick_arr_el(i, bool), [list], bool, 'pick_bool_' + str(i))
        # self.addPrimitive(pick_arr_el(i, str), [list], str, 'pick_str_' + str(i))

      # add ability to set the value of the registers, where the registers are the last 5 elements of the input array
      # set_el_5(x, 2) - means set the 5th element(starting from 0) in the input array to 2
      # It can only set values of registers. - for [input_1, input_2, reg_1, reg_2], the only possible
      # operators are set_el_2 and set_el_3.
      for i in range(self.list_length, self.list_length + REGISTERS_COUNT):
        self.addPrimitive(set_arr_el(i), [list, object], list, 'set_' + str(i))
        # self.addPrimitive(set_arr_el(i), [list, bool], list, 'set_bool_' + str(i))
        # self.addPrimitive(set_arr_el(i), [list, str], list, 'set_str_' + str(i))

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
        # self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=10)
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.toolbox.register("evaluate", self.individual_fitness_eval_func)

        # makes the GP run in multiprocessing
        # if __name__ == '__main__':
        #   self.toolbox.register("map", futures.map)

        self.addSelectionTool()
        self.addCrossOverTool()
        self.addMutationTool()

        # required for either semantic mutation or for semantic crossover
        self._addCxSemanticLFFunc()

        # Then, we decorate the mate and mutate method to limit the height of generated individuals. 
        # This is done to avoid an important draw back of genetic programming : bloat. 
        # Koza in his book on genetic programming suggest to use a max depth of 17.
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

    def addMutationTool(self):
      self.toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=6)

      if self.mut_tool == 'mutShrink':
        self.toolbox.register("mutate", gp.mutShrink)
      elif self.mut_tool == 'mutUniform':
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
      elif self.mut_tool == 'mutNodeReplacement':
        self.toolbox.register("mutate", gp.mutNodeReplacement, pset=self.pset)
      elif self.mut_tool == 'mutInsert':
        self.toolbox.register("mutate", gp.mutInsert, pset=self.pset)
      elif self.mut_tool == 'mutSemantic':
        self.toolbox.register("mutate", gp.mutSemantic, gen_func=gp.genHalfAndHalf, pset=self.pset, min=2, max=6, ms=None)

    def _addCxSemanticLFFunc(self):
        def lf(x): 
          return 1 / (1 + exp(-x))

        self.pset.addPrimitive(lf, [float], float, name="lf")

    def addCrossOverTool(self):
      if self.cx_tool == 'cxOnePointLeafBiased':
        self.toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
      elif self.cx_tool == 'cxOnePoint':
        self.toolbox.register("mate", gp.cxOnePoint)
      elif self.cx_tool == 'cxSemantic':
        self.toolbox.register("mate", gp.cxSemantic, gen_func=gp.genHalfAndHalf, pset=self.pset, min=2, max=6)

    def addSelectionTool(self):
      if self.selection == 'sel_tourn':
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournsize)
      # elif self.selection == 'sel_tourn_dcd':
      #   self.toolbox.register("select", tools.selTournamentDCD)
      elif self.selection == 'sel_tourn_double':
        self.toolbox.register("select", tools.selDoubleTournament, fitness_size=self.tournsize, parsimony_size=self.tournparssize, fitness_first=True)
      elif self.selection == 'sel_random':
        self.toolbox.register("select", tools.selRandom)
      elif self.selection == 'sel_best':
        self.toolbox.register("select", tools.selBest)
      elif self.selection == 'sel_worst':
        self.toolbox.register("select", tools.selWorst)
      elif self.selection == 'sel_stoch':
        self.toolbox.register("select", tools.selStochasticUniversalSampling)
      elif self.selection == 'sel_lexicase':
        self.toolbox.register("select", tools.selLexicase)
      elif self.selection == 'sel_eps_lexicase':
        self.toolbox.register("select", tools.selEpsilonLexicase, epsilon=0.5)
      elif self.selection == 'sel_auto_eps_lexicase':
        self.toolbox.register("select", tools.selAutomaticEpsilonLexicase)


    def score(self, x, y):
      return self.individual_fitness_eval_func(
        individual=self.get_best_tree(),
        test_x_y_list=x
      )


    def add_stats(self):
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        # self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self.mstats = Stats(fitness=stats_fit, size=stats_size)
        self.mstats.register("avg", numpy.mean)
        self.mstats.register("std", numpy.std)
        self.mstats.register("min", numpy.min)
        self.mstats.register("max", numpy.max)

    def run(self):
      population = self.toolbox.population(n=self.population_size)
      population, logbook = self.population_gen_func(population, self)

      return logbook

    def get_best_tree(self):
        return self.hof[0]

    def get_best_tree_expression(self):
        return self.toolbox.compile(expr=self.get_best_tree())