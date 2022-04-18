import random, string

def generate_random_string(length = 3):
  letters = [string.ascii_lowercase[random.randint(0, 25)] for i in range(0, length)]
  
  return ''.join(letters)