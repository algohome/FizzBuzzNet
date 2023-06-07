import csv
import numpy as np

def _cast_list_to_type(l:list, t):
    return [t(x) for x in l]

def _validate_array_dtype(arr):
    pass 

def _validate_tensor_dtype(arr):
    pass 

def load_fizzbuzz_csv():
    x_data = []
    y_data = []
    with open('../data/fizzbuzzdata.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x_data.append(row[0])
            y_data.append(row[1])
    return x_data, y_data

def fizzbuzz_to_formatted_array(y_data:list):
    output_arr = []
    for val in y_data:
        out_val = []
        if "Fizz" in val:
            out_val.append(1)
        else:
            out_val.append(0)
        
        if "Buzz" in val:
            out_val.append(1)
        else:
            out_val.append(0)
        
        output_arr.append(out_val)
    return np.array(output_arr)

def get_model_fizzbuzz(model, n, conf_thresh = 0.5):
    res = model.predict(np.array([n])).flatten()
    formatted_output = ""
    if res[0] > conf_thresh:
        formatted_output += "Fizz"
    if res[1] > conf_thresh:
        formatted_output += "Buzz"
    if formatted_output == "":
        formatted_output += str(n)
    return formatted_output

# Returns one-hot array, ['Fizz', 'Buzz']
def FizzBuzzSolver(n):
    output_arr = []
    if n % 3 == 0:
        output_arr.append(1)
    else:
        output_arr.append(0)
    
    if n % 5 == 0:
        output_arr.append(1)
    else:
        output_arr.append(0)

    return output_arr

def evaluate_model(model, n):
    total_correct = 0
    for i in range(n):
        res = str(get_model_fizzbuzz(model, np.array([i])))
        true_res = FizzBuzzSolver(i)
        true_str = ""
        if true_res[0] == 1:
            true_str += "Fizz"
        if true_res[1] == 1:
            true_str += "Buzz"
        if true_str == "":
            if not "Fizz" in res and not "Buzz" in res:
                total_correct += 1
        total_correct += res == true_str 
    return total_correct/n

def x_data_reformatter(n, input_dim):
    bin_n = bin(n)
    str_n = bin_n[2:]
    if len(str_n) < input_dim:
        str_n = '0'*(input_dim-len(str_n)) + str_n 
    return [int(char) for char in str_n]

def evaluate_binary_model(model, n):
    total_correct = 0
    for i in range(n):
        res = str(get_model_fizzbuzz(model, np.array([x_data_reformatter(i, 30)])))
        true_res = FizzBuzzSolver(i)
        true_str = ""
        if true_res[0] == 1:
            true_str += "Fizz"
        if true_res[1] == 1:
            true_str += "Buzz"
        if true_str == "":
            if not "Fizz" in res and not "Buzz" in res:
                total_correct += 1
        total_correct += res == true_str 
    return total_correct/n