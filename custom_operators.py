import itertools

def protectedDivision(left, right):
    try:
        return left / right
    except:
        return 1

def safe_binary_operation(func, default_value):
    
    def op(arg1, arg2):
        res = None
        try:
            arg1 + 1
            arg2 + 1
            res = func(arg1, arg2)
        except ValueError as e:
            res = default_value
        return res

    return op

def _safe_operation(arg, func, default_value):
    try:
        return func(arg)
    except:
        return default_value

def sum_list_elements(index1, index2):
    return lambda args_array: _safe_operation(args_array, lambda args: args[index1] + args[index2], 0)

def subtract_list_elements(index1, index2):
    return lambda args_array: _safe_operation(args_array, lambda args: args[index1] - args[index2], 0)

def subtract_list_elements(index1, index2):
    return lambda args_array: _safe_operation(args_array, lambda args: args[index1] - args[index2], 0)

def generate_index_combinations(args_length, combination_length):
    arg_indexes = list(range(args_length))
    return list(itertools.combinations(arg_indexes, combination_length))

# float, bool can be converted to str
# float, str can be converted to bool
# but str may not be convertable to float
def pick_arr_el(index, output_type):
# def pick_arr_el(index):
    def pick(array):
        res = array[index]

        return output_type(res)

    return pick

def set_arr_el(index):
    def set_value(array, value):
        array[index] = value

        return array

    return set_value
