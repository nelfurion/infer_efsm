import random
from ..generator import TraceGenerator

tg = TraceGenerator('x_squared')
tg.generate(
  'enter(1), receive(1)',
  {
    'enter': lambda prev_inputs, prev_outputs: random.randint(1, 300),
    'enter_output': lambda prev_inputs, prev_outputs: '',
    'receive': lambda prev_inputs, prev_outputs: '',
    'receive_output': lambda prev_inputs, prev_outputs: prev_inputs[0] * prev_inputs[0]
  }
)
tg.write_to_file()