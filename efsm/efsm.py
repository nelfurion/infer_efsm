class EFSM:
  # e.g. 
  # input_parameters_list = [0, 1, 2]
  # therefore each state will be initialized with functions that can work with input_parameters_list[0:2]
  def __init__(self, input_parameters_list):
    self.input_parameters_list = input_parameters_list
  
  def set_states(self, states):
    self.states = states

  def set_transitions(self, transitions):
    self.transitions = transitions
