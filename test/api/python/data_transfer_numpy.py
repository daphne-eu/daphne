import numpy as np
import pandas as pd
from daphne.context.daphne_context import DaphneContext

dctx = DaphneContext()

test_cases = [
    (np.array([1.0, 2.0, 3.0], dtype=np.float64).reshape(-1, 1), "float64_1d"),
    (np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float64).reshape(-1, 1), "float64_1d_longer"),
    #(np.array(["apple", "banana", "cherry"], dtype=str).reshape(-1, 1), "string_fruits"),
]

test_cases_string = [
    # 1D arrays
    (np.array(["apple", "banana", "cherry"], dtype=str).reshape(-1, 1), "string_fruits"),
    #(np.array(["apple", "banana", "cherry", "date"], dtype=object).reshape(-1, 1), "string_fruits_longer"),
    #(np.array(["apple", "banana", "cherry", "date", "elderberry"], dtype=object).reshape(-1, 1), "string_fruits_even_longer"),
]

test_cases_string_pandas = [
    # Pandas Series
    (pd.Series(["apple", "banana", "cherry"], dtype=str), "string_series"),
    #(pd.Series(["dog", "elephant", "fox", "giraffe"], dtype=str), "string_series_longer"),
    
    # Pandas DataFrames
    #(pd.DataFrame({"col1": ["red", "green", "blue"], "col2": ["circle", "square", "triangle"]}), "string_df"),
    #(pd.DataFrame({"col1": ["cat", "dog"], "col2": ["fish", "bird"], "col3": ["hamster", "rabbit"]}), "string_df_wider"),
    #(pd.DataFrame({"col1": ["one", "two", "three"], "col2": ["four", "five", "six"], "col3": ["seven", "eight", "nine"], "col4": ["ten", "eleven", "twelve"]}), "string_df_taller"),
]

for X, name in test_cases:
    try:
        result = dctx.from_numpy(X, shared_memory=True).print().compute()
        print(f"Result for {name}: {result}")
    except Exception as e:
        print(f"Error for {name}: {e}")

for X, name in test_cases_string:
    try:
        result = dctx.from_numpy2(X, shared_memory=True).print().compute()
        print(f"Result for {name}: {result}")
    except Exception as e:
        print(f"Error for {name}: {e}")

for X, name in test_cases_string_pandas:
    try:
        result = dctx.from_pandas(X, shared_memory=True).print().compute()
        print(f"Result for {name}: {result}")
    except Exception as e:
        print(f"Error for {name}: {e}")