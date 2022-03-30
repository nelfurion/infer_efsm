import itertools

def protectedDivision(left, right):
    try:
        return left / right
    except:
        return 1

def safe_binary_operation(func, default_value):
    
    def op(arg1, arg2):
        # print(type(arg1), ' ', type(arg2))
        res = None
        try:
            arg1 + 1
            arg2 + 1
            res = func(arg1, arg2)
        except ValueError as e:
            # print(e)
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
# def pick_arr_el(index, output_type):
def pick_arr_el(index):
    def pick(array):
        res = array[index]
        # return output_type(res)
        return res

    return pick

def set_arr_el(index):
    def set_value(array, value):
        # if value > 1:
        #     print('SETTING ', index, ' to ', value)
        #     print('array before: ', array)
        
        array[index] = value

        # if value > 1:
        #     print('array after: ', array)

        return array

    return set_value

# inputs[-1] is a list that holds indexes of elements to be used as outputs
def select_element_index_to_use_in_output(output_el_index):
    def select_output_el_index(array):
        try: 
            array[-1].append(output_el_index)
            return array
        except:
            return 0

    return select_output_el_index

# inputs[-2] is a list that holds indexes of elements to be used as output conditions
def select_element_index_to_use_as_output_condition(output_cond_el_index):
    def select(array):
        try: 
            array[-2].append(output_cond_el_index)
            return array
        except:
            return 0

    return select

def cycle_conditions_and_outputs(inputs, default_value):
    # print('cycle_conditions_and_outputs')
    # print(inputs)
    conditions = inputs[-2]
    outputs = inputs[-1]

    # print(conditions, outputs)
    if len(conditions) == 0 or len(outputs) == 0:
        return default_value
    else:
        for index in range(len(outputs)):
            if index in range(len(conditions)):
                condition = inputs[index]
                if condition:
                    return outputs[index]
            # if we have more outputs than conditions and were not able to prove either output
            else:
                break

    
    return default_value
    
