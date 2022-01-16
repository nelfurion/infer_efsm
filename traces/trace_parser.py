from traces.trace_generator import TraceGenerator


class TraceParser:
  pass

class TraceParser:
  def __init__(self, file_path: str) -> TraceParser:
    self.file_path = file_path
  
  def parse(self) -> list:
    '''
      Returns a dict with the events and their inputs and outputs.
      [max inputs count, dict]

      dict: {
        event_name: [
          ([inputs], output),
          ([inputs], output),
          ([inputs], output)
        ]
      }
    '''
    all_events = {}
    max_inputs_count = 0
    with open(self.file_path, 'r') as trace_file:
      for trace in trace_file:
        trace_events = trace.split(',')
        # event_names = [ event.split('/')[0].split('(')[0].strip() for event in events]
        prev_inputs = []
        for event in trace_events:
          event_name = event.split('/')[0].split('(')[0].strip()
          event_input = self._try_parse_int(event.split('(')[1].split(')')[0])
          event_output = self._try_parse_int(event.split('[')[1].split(']')[0])
          all_events[event_name] = all_events[event_name] if event_name in all_events else []


          all_events[event_name].append(
            (prev_inputs + [event_input], event_output)
          )

          prev_inputs.append(event_input)
          max_inputs_count = max((max_inputs_count, len(prev_inputs)))

    return [max_inputs_count, all_events]

  def _try_parse_int(self, value):
    try:
      return int(value)
    except ValueError:
      return value

# tp = TraceParser('./traces/x_squared/traces_2191')
# print(tp.parse())