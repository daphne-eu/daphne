import pandas as pd
import numpy as np
from daphne.context.daphne_context import DaphneContext

dctx = DaphneContext()

# Test cases for Pandas Series
series_test_cases = [
    # Simple Series
    (pd.Series([1.0, 2.0, 3.0]), "float64_series"),
    (pd.Series([1, 2, 3], dtype=np.int32), "int32_series"),
    
    # Series with different shapes
    (pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), "float64_series_longer"),
    
    # Edge cases
    (pd.Series([], dtype=np.float64), "float64_empty_series"),
    
    # Series with categorical data
    (pd.Series(pd.Categorical(["a", "b", "c"])), "categorical_series"),
]

# Test cases for Pandas DataFrames
dataframe_test_cases = [
    # Simple DataFrames
    (pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}), "float64_dataframe"),
    (pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, dtype=np.int32), "int32_dataframe"),
    
    # DataFrames with different shapes
    (pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]}), "float64_dataframe_wider"),
    (pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [5.0, 6.0, 7.0, 8.0]}), "float64_dataframe_taller"),
    
    # Edge cases
    (pd.DataFrame({"A": [], "B": []}, dtype=np.float64), "float64_empty_dataframe"),
]

# Testing Pandas Series
for series, name in series_test_cases:
    try:
        series_daphne = dctx.from_pandas(series, shared_memory=True)
        
        series_daphne.print().compute(type="shared memory")
        
    except Exception as e:
        print(f"Error for {name}: {e}")

# Testing Pandas DataFrames
for df, name in dataframe_test_cases:
    try:    
        # Transfer Pandas DataFrame to DAPHNE
        df_daphne = dctx.from_pandas(df, shared_memory=True)
        
        # Print the Daphne frame
        df_daphne.print().compute(type="shared memory")
    
    except Exception as e:
        print(f"Error for {name}: {e}")