import random
import string
from ..trace_generator import TraceGenerator

def generate_random_string(length = 3):
  letters = [string.ascii_lowercase[random.randint(0, 25)] for i in range(0, length)]
  
  return ''.join(letters)

tg = TraceGenerator('vending_machine')
tg.generate(
  'select("tea"), coin(50)/[50], coin(5950)/[6000], vend()/["tea"]',
  {
    'select': lambda prev_inputs, prev_outputs: f'"{generate_random_string()}"',
    'select_output': lambda prev_inputs, prev_outputs: '',
    'coin': lambda prev_inputs, prev_outputs: 1000 - prev_inputs[1] if len(prev_inputs) > 1 else random.randint(0, 999),
    'coin_output': lambda prev_inputs, prev_outputs: prev_inputs[1] if len(prev_inputs) == 2 else prev_inputs[1] + prev_inputs[2],
    'vend': lambda prev_inputs, prev_outputs: '',
    'vend_output': lambda prev_inputs, prev_outputs: prev_inputs[0]
  },
  6000
)
tg.write_to_file()
