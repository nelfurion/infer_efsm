import random
import string
from ..generator import TraceGenerator

def generate_random_string(length = 3):
  letters = [string.ascii_lowercase[random.randint(0, 25)] for i in range(0, length)]
  
  return ''.join(letters)

def get_random_pin():
  pins = ['1111', '1234', '2323', '0000']

  return pins[random.randint(0, 3)]

ACCOUNT_AMOUNT = 100
ACCOUNT_PIN = '1111'

tg = TraceGenerator('simple_atm_withdraw_only')
tg.generate(
  'select_operation("withdraw"), select_amount(150)/[], enter_pin(1111)/[], check_account()/["not enough money in account"]',
  {
    'select_operation': lambda prev_inputs, prev_outputs: '"withdraw"',
    'select_operation_output': lambda prev_inputs, prev_outputs: '',
    'select_amount': lambda prev_inputs, prev_outputs: random.randint(1, 200),
    'select_amount_output': lambda prev_inputs, prev_outputs: '',
    'enter_pin': lambda prev_inputs, prev_outputs: get_random_pin(),
    'enter_pin_output': lambda prev_inputs, prev_outputs: '',
    'check_account': lambda prev_inputs, prev_outputs: '',
    # 'check_account_output': lambda prev_inputs, prev_outputs: '"not enough money in account"' if prev_inputs[1] > ACCOUNT_AMOUNT else ,
  }
)
tg.write_to_file()
