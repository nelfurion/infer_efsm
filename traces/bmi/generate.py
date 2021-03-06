import random
from ..trace_generator import TraceGenerator

def bmi(height, weight):
  bmi_value = weight / ((height/100) ** 2)

  if bmi_value >= 18.5 and bmi_value <= 24.9:
    return 'healthy'
  elif bmi_value < 18.5:
    return 'underweight'
  else:
    return 'overweight'
    

tg = TraceGenerator('bmi')
tg.generate(
  'height(50), weight(50), bmi()/[100]',
  {
    # in centimeters
    'height': lambda prev_inputs, prev_outputs: random.randint(150, 230),
    'height_output': lambda prev_inputs, prev_outputs: '',
    # in kilograms
    'weight': lambda prev_inputs, prev_outputs: random.randint(50, 100),
    'weight_output': lambda prev_inputs, prev_outputs: '',
    'bmi': lambda prev_inputs, prev_outputs: '',
    'bmi_output': lambda prev_inputs, prev_outputs: bmi(prev_inputs[0], prev_inputs[1]),
  },
  5000
)
tg.write_to_file()


# import random
# import string
# from ..generator import TraceGenerator

# def generate_random_string(length = 3):
#   letters = [string.ascii_lowercase[random.randint(0, 25)] for i in range(0, length)]
  
#   return ''.join(letters)

# tg = TraceGenerator('vending_machine')
# tg.generate(
#   'select("tea"), coin(50)/[50], coin(50)/[100], vend()/["tea"]',
#   {
#     'select': lambda prev_inputs, prev_outputs: f'"{generate_random_string()}"',
#     'select_output': lambda prev_inputs, prev_outputs: '',
#     'coin': lambda prev_inputs, prev_outputs: 100 - prev_inputs[1] if len(prev_inputs) > 1 else random.randint(0, 50),
#     'coin_output': lambda prev_inputs, prev_outputs: prev_inputs[1] if len(prev_inputs) == 2 else prev_inputs[1] + prev_inputs[2],
#     'vend': lambda prev_inputs, prev_outputs: '',
#     'vend_output': lambda prev_inputs, prev_outputs: prev_inputs[0]
#   }
# )
# tg.write_to_file()
