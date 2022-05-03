import itertools
import random
import math

from deap import tools
from operator import attrgetter


def protectedDivision(left, right):
    try:
        return left / right
    except:
        return 1

def safe_binary_operation(func, default_value):
    
    def op(arg1, arg2):
        res = None
        try:
            arg1 + 1
            arg2 + 1
            res = func(arg1, arg2)
        except ValueError as e:
            res = default_value
        return res

    return op

def _safe_operation(arg, func, default_value):
    try:
        return func(arg)
    except:
        return default_value

def sum_list_elements(index1, index2):
    return lambda args_array: _safe_operation(args_array, lambda args: args[index1] + args[index2], 0)

def subtract_list_elements(index1, index2):
    return lambda args_array: _safe_operation(args_array, lambda args: args[index1] - args[index2], 0)

def subtract_list_elements(index1, index2):
    return lambda args_array: _safe_operation(args_array, lambda args: args[index1] - args[index2], 0)

def generate_index_combinations(args_length, combination_length):
    arg_indexes = list(range(args_length))
    return list(itertools.combinations(arg_indexes, combination_length))

# float, bool can be converted to str
# float, str can be converted to bool
# but str may not be convertable to float
def pick_arr_el(index, output_type):
# def pick_arr_el(index):
    def pick(array):
        res = array[index]

        return output_type(res)

    return pick

def set_arr_el(index):
    def set_value(array, value):
        array[index] = value

        return array

    return set_value


def _string_difference(str1, str2):
    return len(set(str(str1)).symmetric_difference(set(str(str2))))

def selTournamentDifferent(individuals, k, tournsize, fit_attr="fitness"):
    chosen_best = tools.selTournament(individuals, math.ceil(k/2), tournsize, fit_attr=fit_attr)
    chosen_different = []
    for i in range(math.floor(k/2)):
        aspirants = tools.selRandom(individuals, tournsize)
        differences = [{ 
                'individual': asp, 
                'diff_length':_string_difference(asp, chosen_best[i]) 
            } 
            for asp in aspirants
        ]

        chosen_individual = max(differences, key=lambda x: x['diff_length'])
        chosen_different.append(chosen_individual['individual'])

    return chosen_best + chosen_different

def selTournBestAndWorst(individuals, k, tournsize, fit_attr="fitness"):
    chosen_best = tools.selTournament(individuals, math.ceil(k/2), tournsize, fit_attr=fit_attr)
    chosen_worst = []
    for i in range(math.floor(k/2)):
        aspirants = tools.selRandom(individuals, tournsize)
        chosen_worst.append(min(aspirants, key=attrgetter(fit_attr)))

    return chosen_best + chosen_worst



# def selRandom(individuals, k):
#     """Select *k* individuals at random from the input *individuals* with
#     replacement. The list returned contains references to the input
#     *individuals*.
#     :param individuals: A list of individuals to select from.
#     :param k: The number of individuals to select.
#     :returns: A list of selected individuals.
#     This function uses the :func:`~random.choice` function from the
#     python base :mod:`random` module.
#     """
#     return [random.choice(individuals) for i in xrange(k)]


# for i in range(math.floor(k/2)):
#      aspirants = ind
#      differences = [{ 
#                  'individual': asp, 
#                  'diff_length':_string_difference(asp, chosen_best[i]) 
#              } 
#              for asp in aspirants
#      ]
#      print('differences:')
#      print(differences)
#      chosen_individual = max(differences, key='diff_length')
#      print('chosen ind', chosen_individual)
#      chosen_different.append(chosen_individual['individual'])
