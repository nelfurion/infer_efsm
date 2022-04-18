import operator

import math, numpy
import typing

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from custom_operators import cycle_conditions_and_outputs, select_element_index_to_use_in_output, select_element_index_to_use_as_output_condition, generate_index_combinations, pick_arr_el, protectedDivision, safe_binary_operation, set_arr_el, subtract_list_elements, sum_list_elements
from plot import plot_tree, plot_two, plot_two_2
from traces.trace_parser import TraceParser

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
    def __init__(self, population_size, hof_size, input_types, output_type, list_length, generations_count, individual_fitness_eval_func = None) -> None:
        self.pset = gp.PrimitiveSetTyped("MAIN", input_types, output_type)
        self.output_type = output_type
        self.toolbox = base.Toolbox()
        self.population_size = population_size
        self.hof = tools.HallOfFame(hof_size)

        self.list_length = list_length
        self.generations_count = generations_count

        self.individual_fitness_eval_func = individual_fitness_eval_func

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
        generations_count=setup['generations_count'],
        individual_fitness_eval_func=setup['individual_fitness_eval_func'] if 'individual_fitness_eval_func' in setup.keys() else None
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
      # self.addSelectOutputElementPrimitives()
      # self.addSelectOutputConditionPrimitives()

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

      # self.addPrimitive(cycle_conditions_and_outputs, [list, self.output_type], self.output_type, 'loop_conditions_outputs')

    # def addSelectOutputElementPrimitives(self):
    #   for i in range(self.list_length, self.list_length + REGISTERS_COUNT - 2):
    #     self.addPrimitive(select_element_index_to_use_in_output(i), [list], list, 'select_el_for_output_' + str(i))

    # def addSelectOutputConditionPrimitives(self):
    #   for i in range(self.list_length, self.list_length + REGISTERS_COUNT - 2):
    #     self.addPrimitive(select_element_index_to_use_as_output_condition(i), [list], list, 'select_el_for_output_condition_' + str(i))

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

        self.toolbox.register("evaluate", self.individual_fitness_eval_func or self.eval_mean_squared_error)
        self.toolbox.register("select", tools.selTournament, tournsize=10)
        self.toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
        # self.toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=10)
        self.toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

        # Then, we decorate the mate and mutate method to limit the height of generated individuals. 
        # This is done to avoid an important draw back of genetic programming : bloat. 
        # Koza in his book on genetic programming suggest to use a max depth of 17.
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    def eval_mean_squared_error(self, individual, test_x_y_list=None, y_only_list=None):
        # Transform the tree expression in a callable function
        tree_expression = self.toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression
        # and the real function : x**4 + x**3 + x**2 + x

        # print('x_y_list: ', x_y_list)
        # print('self.target_list: ', self.target_list)

        squared_errors = []
        for x_y in (test_x_y_list or self.target_list):
          # print(func(x_y[0]))
          # print(x_y[1])
          try:
            # EDIT THIS
            # THIS IS JUST TEST IMPLEMENTATION OF RUNNING A FUNCTION MULTIPLE TIMES WITH A SINGLE PARAMETER

            # this is the coin event in the vending machine
            # the params for the coin event are from indexes 1 until the end of the array
            # FIX THIS PART
            params = x_y[0][1:]
            # print('params: ', params)

            registers = [0, 0, 0, 0, 0]
            output_condition_elements = []
            output_elements = []

            tree_expression_result = None
            # lets try to call the tree multiple times with a single parameter each time
            for param in params:

              # print('calling tree with param: ', param)
              # print('registers: ', registers)
              # pass the param in a list, so that we don't have to change the pick_array_element implementation
              # we will hardcode it to only work for the 0th index of the input, and also for 5 more indexes for 
              # custom registers.
              param_and_registers = [param] + registers + [output_condition_elements, output_elements]
              # print('params list: ', params, ' CALLING WITH: ', param_and_registers)
              # print('TREE: ', individual)
              tree_expression_result = tree_expression(param_and_registers)
              registers = param_and_registers[-5:]
            # tree_expression_result = tree_expression(x_y[0]) // this is old code

            # only use the last tree expression result from above
            squared_error = (tree_expression_result - x_y[1]) ** 2
            squared_errors.append(squared_error)

            # print('------------------')
          except TypeError as e: # if the tree is just: x , then we have array - integer
            print(e)
            print(individual)
            pass
        # squared_errors = () for x_y in self.target_list)

        return math.fsum(squared_errors) / len(squared_errors) if len(squared_errors) else 20000,
    
    def score(self, x, y):
      return self.individual_fitness_eval_func(
        individual=self.get_best_tree(),
        test_x_y_list=x
      )

      # return self.best_tree_score
      # return self.eval_mean_squared_error(
      #   individual=self.get_best_tree(),
      #   x_y_list=x
      # )


    def add_stats(self):
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self.mstats.register("avg", numpy.mean)
        self.mstats.register("std", numpy.std)
        self.mstats.register("min", numpy.min)
        self.mstats.register("max", numpy.max)

    def run(self):
      population = self.toolbox.population(n=self.population_size)
      population, log = self.population_gen_func(population, self)

    def get_best_tree(self):
        return self.hof[0]

    def get_best_tree_expression(self):
        return self.toolbox.compile(expr=self.get_best_tree())