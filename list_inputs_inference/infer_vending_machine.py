import operator

from deap import algorithms

from custom_operators import protectedDivision, safe_binary_operation
from plot import plot_tree, plot_two_2, plot_3d
from traces.trace_parser import TraceParser

from gp_algorithm import GPListInputAlgorithm

tp = TraceParser('./traces/vending_machine/traces_3855')

event_args_length, events = tp.parse()
event_args_length = len(events['coin'][0][0])

# print(event_args_length)
# print(events['vend'][0][0])
# print(len(events['vend']))
# print(len(events['coin']))
print(events['coin'][0:5])

# gpa.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
gp_setup = {
  'population_size': 300,
  'hall_of_fame_size': 2,
  'input_list_length': 1, # hardcoding it to only accept a single argument # event_args_length,
  'output_type': float,
  'generations_count': 1000,
  'primitives': [
    [safe_binary_operation(operator.add, 0), [float, float], float, 'add'],
    [safe_binary_operation(operator.sub, 0), [float, float], float, 'sub'],
    [safe_binary_operation(operator.mul, 0), [float, float], float, 'mul'],
    [protectedDivision, [float, float], float, 'div']
  ],
  'terminals':[
    [0, float]
  ],
  'target': events['coin'],
  # 'population_generation_func': lambda population, gpa: algorithms.eaMuCommaLambda(
  #   population, 
  #   gpa.toolbox, 
  #   10, 20, 0.2, 0.1,
  #   gpa.generations_count,
  #   stats=gpa.mstats,
  #   halloffame=gpa.hof,
  #   verbose=True
  # )
  # 'population_generation_func': lambda population, gpa: 
  #   algorithms.eaSimple(
  #   population, 
  #   gpa.toolbox, 
  #   cxpb=0.5, 
  #   mutpb=0.2, 
  #   ngen=gpa.generations_count, 
  #   stats=gpa.mstats, 
  #   halloffame=gpa.hof, 
  #   verbose=True
  # ),
  'population_generation_func': lambda population, gpa: algorithms.eaMuPlusLambda(
    population,
    gpa.toolbox,
    10, 20, 0.2, 0.1,
    gpa.generations_count,
    stats=gpa.mstats,
    halloffame=gpa.hof,
    verbose=True
  )
}

gpa = GPListInputAlgorithm.create(gp_setup)
gpa.run()

best_tree = gpa.get_best_tree()
plot_tree(best_tree)
best_tree_score = gpa.eval_mean_squared_error(best_tree)
print("Best Tree Syntax: ", str(best_tree))
print("Best Tree Score: ", best_tree_score)

best_tree_expression = gpa.get_best_tree_expression()
ideal_func = lambda args: args[0] * args[0]
func_args = list(map(lambda x: x[0], events['coin']))




# print('func args:', func_args)
# plot_two_2(func_args, ideal_func, best_tree_expression, 0)
# plot_3d(range(0, 1000), range(0, 1000), list(map(lambda x: [x, x], range(0, 1000))))


















# # [
# #   gpa.addPrimitive(
# #     sum_list_elements(indexes[0], indexes[1]),
# #     1
# #   ) for indexes in generate_index_combinations(2, 2)
# # ]

# # [
# #   gpa.addPrimitive(
# #     subtract_list_elements(indexes[0], indexes[1]),
# #     1
# #   ) for indexes in generate_index_combinations(2, 2)
# # ]

# # gpa.addPrimitive(operator.neg, 1)
# # gpa.addPrimitive(math.cos, 1)
# # gpa.addPrimitive(math.sin, 1)