import numpy as np
import pandas as pd
import re

def apply_map_function(input_file, rows, cols, func, varName):
    arg_array = pd.read_csv(input_file, header=None).values.reshape(rows, cols)

    # Eval Approach
    # res_array = eval(func.replace(varName, 'arg_array'))

    # Lambdify Approach
    # x = symbols(varName)
    # func = lambdify(x, func, modules=["numpy"])
    # res_array = func(arg_array)

    # Exec and Re Approach for full functions defined
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

    pd.DataFrame(res_array).to_csv("output.csv", index=False, header=False)