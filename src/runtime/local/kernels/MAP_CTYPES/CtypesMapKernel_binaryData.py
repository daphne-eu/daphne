import numpy as np
import re

def apply_map_function(input_file, output_file, rows, cols, func, varName):
    # Read input data from binary file
    arg_array = np.fromfile(input_file, dtype=np.float64).reshape(rows, cols)
    
    # Process the data
    match = re.search(r'def (\w+)', func)
    if match:
        try:
            exec(func)
            func_name = match.groups()[0]
            func_obj = locals().get(func_name)
            if func_obj:
                res_array = np.vectorize(func_obj)(arg_array)
            else:
                print(f"Function '{func_name}' not found.")
        except Exception as e:
            print(f"Failed to execute function: {str(e)}")
    else:
        print("No function name found")

    # Write result to binary file
    res_array.tofile(output_file)