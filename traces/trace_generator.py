import os
import random
import errno

class TraceGenerator:
    pass

class TraceGenerator:
  def __init__(self, traces_name) -> TraceGenerator:
    self.traces_name = traces_name
    self.trace_file_path = f'./traces/{traces_name}/traces_{random.randint(1, 10000)}'
    self.trace_file_dir = f'./traces/{traces_name}'
    self.traces = []

  def generate(self, trace_template: str, event_parameter_generators: hash, count = 200) -> list:
    '''
      Generates a list of traces and writes them to a file.
      If the EFSM is not required to go through all states each time, then call
      generate once for each possible trace_template.

      Parameters:
        trace_template: String example of a trace

        event_parameter_generators: Hash where the keys are the event names
          and the values are the event name inputs.
          If the key ends with _output the generator is used for return values.
          The return values generator should accept a list which will be populated with the 
          event input parameters.
    '''
    events = trace_template.split(',')
    event_names = [ event.split('/')[0].split('(')[0].strip() for event in events]

    self.traces = []
    for i in range(0, count):
      previous_inputs = []
      previous_outputs = []
      for event_name in event_names:
        input_value = event_parameter_generators[event_name](previous_inputs, previous_outputs)
        previous_inputs.append(input_value)
        output_value = event_parameter_generators[f'{event_name}_output'](previous_inputs, previous_outputs)
        previous_outputs.append(output_value)

      trace_as_list = [
        f'{event_name}({previous_inputs[j]})/[{previous_outputs[j]}]' for j, event_name in enumerate(event_names)
      ]

      self.traces.append(', '.join(trace_as_list))

    return self.traces

  def write_to_file(self, mode = 'w+') -> None:
    try:
        os.makedirs(self.trace_file_dir)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(self.trace_file_dir):
            pass
        else: raise

    with open(self.trace_file_path, mode) as trace_file:
      for trace in self.traces:
        trace_file.write(trace + "\n")