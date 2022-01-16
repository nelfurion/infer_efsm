import random
from ..generator import TraceGenerator

tg = TraceGenerator('cashier')
tg.generate(
  'select("snack"), pay(50)/[50], pay(50)/[100], serve_snack()/["snack"]',
  {
    'select': lambda prev_inputs, prev_outputs: random.randint(1, 300),
    'select_output': lambda prev_inputs, prev_outputs: '',
    'receive': lambda prev_inputs, prev_outputs: '',
    'receive_output': lambda prev_inputs, prev_outputs: prev_inputs[0] * prev_inputs[0]
  }
)
tg.write_to_file()