import random
from ..trace_generator import TraceGenerator

tg = TraceGenerator('is_even')
tg.generate(
  'is_even(1)/["no"]',
  {
    'is_even': lambda prev_inputs, prev_outputs: random.randint(1, 100000),
    'is_even_output': lambda prev_inputs, prev_outputs: 'yes' if prev_inputs[0] % 2 == 0 else 'no',
  },
  5000
)
tg.write_to_file()