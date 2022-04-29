# common_params_grid = {
#   'mu': [5, 10, 100], #[5, 10, 100],
#   'lmbda': [10, 50, 100], #[10, 50, 100], #[10, 20],
#   'cxpb': [0.1, 0.5, 0.8], #[0.1, 0.5, 0.8], #[0.1, 0.2],
#   'mutpb': [0.1, 0.15, 0.2], #[0.1, 0.15, 0.2], # [0.1, 0.2],
#   'gcount': [50, 500, 1000],#[50, 1000],
#   'popsize': [500, 1000, 10000],#[100, 1000, 10000],
#   'cx_tool': ['cxOnePoint', 'cxOnePointLeafBiased', 'cxSemantic'],
#   'mut_tool': ['mutShrink', 'mutUniform', 'mutNodeReplacement', 'mutInsert', 'mutSemantic'],
#   'tournsize': [7],
#   'tournparssize': [1.4],
# }

# MU AND POPSIZE TESTED VALUES
common_params_grid = {
  'mu': [1000], #[5, 10, 100],
  'lmbda': [10], #[10, 50, 100], #[10, 20],
  'cxpb': [0.8], #[0.1, 0.5, 0.8], #[0.1, 0.2],
  'mutpb': [0.15], #[0.1, 0.15, 0.2], # [0.1, 0.2],
  'gcount': [500],#[50, 1000],
  'popsize': [1000],#[100, 1000, 10000],
  'cx_tool': ['cxOnePoint', 'cxOnePointLeafBiased'],
  'mut_tool': ['mutShrink', 'mutUniform'],
  # 'cx_tool': ['cxOnePoint', 'cxOnePointLeafBiased', 'cxSemantic'],
  # 'mut_tool': ['mutShrink', 'mutUniform', 'mutNodeReplacement', 'mutInsert', 'mutSemantic'],
  'tournsize': [7],
  'tournparssize': [1.4],
}


#  DCD TOURNAMENT SELECTION

# selection = 'sel_tourn_dcd'
# for i in range(26,27):
#   dir = './results/vending_machine/' + selection + '/' + str(i) + '/'
#   Path(dir).mkdir(parents=True, exist_ok=True)

#   grid_search = GridSearchCV(
#     estimator=Estimator(dir), 
#     verbose=10,
#     param_grid={
#       'mu': [4, 8],
#       'lmbda': [12], #[10, 20],
#       'cxpb': [0.1], #[0.1, 0.2],
#       'mutpb': [0.1], # [0.1, 0.2],
#       'gcount': [50],
#       'popsize': [100],
#       'selection': [selection],
#     },
#   )

#   log_file = open(dir + 'log.txt',"w")
#   sys.stdout = log_file

#   grid_search.fit(x_y_list, x_y_list)
#   dataframe = pd.DataFrame(grid_search.cv_results_)
#   dataframe.to_csv(dir + 'result' + '_' + str(i) + '.csv')

#   log_file.close()


# DOUBLE TOURNAMENT SELECTION


# selection = 'sel_tourn_double'
# dir =  self.results_function_dir + selection + '/'
# Path(dir).mkdir(parents=True, exist_ok=True)

# params_grid = copy.deepcopy(common_params_grid)

# params_grid['selection'] = [selection]
# params_grid['tree_output_dir'] = [dir]

# self._run_grid_search_cv_and_log(dir, self.function_inputs, self.function_outputs, params_grid)


# print('***************************', file=sys.stderr)
# print('***************************', file=sys.stderr)
# print('***************************', file=sys.stderr)
# print("DOUBLE SELECTION DONE", file=sys.stderr)
# print('***************************', file=sys.stderr)
# print('***************************', file=sys.stderr)
# print('***************************', file=sys.stderr)

# #  RANDOM SELECTION

# selection_operators = ['sel_best', 'sel_stoch', 'sel_lexicase', 'sel_auto_eps_lexicase']
# for selection in selection_operators:
#   dir =  self.results_function_dir + selection + '/'
#   Path(dir).mkdir(parents=True, exist_ok=True)

#   params_grid = copy.deepcopy(common_params_grid)
#   params_grid['selection'] = [selection]
#   params_grid['tree_output_dir'] = [dir]

#   self._run_grid_search_cv_and_log(dir, self.function_inputs, self.function_outputs, params_grid)

#   print('***************************', file=sys.stderr)
#   print('***************************', file=sys.stderr)
#   print('***************************', file=sys.stderr)
#   print(selection + " DONE", file=sys.stderr)
#   print('***************************', file=sys.stderr)
#   print('***************************', file=sys.stderr)
#   print('***************************', file=sys.stderr)