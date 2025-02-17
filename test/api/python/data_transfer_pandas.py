import numpy as np
import pandas as pd
from daphne.context.daphne_context import DaphneContext

dctx = DaphneContext()

# ==== TEST CASES FOR pd.Series ====
series_test_cases = [
    # 1D Series
    (pd.Series([1.0, 2.0, 3.0]), "float64_1d"),
    (pd.Series([4.0, 5.0, 6.0, 7.0]), "float64_1d_longer"),
    
    # Edge cases
    (pd.Series([np.nan, np.nan, np.nan]), "float64_nan"),
    (pd.Series([np.inf, -np.inf, np.inf]), "float64_inf"),
    (pd.Series([-1.0, -2.0, -3.0]), "float64_negative"),
    
    # Small and Large Numbers
    (pd.Series([1e-10, 2e-10, 3e-10]), "float64_small"),
    (pd.Series([1e10, 2e10, 3e10]), "float64_large"),
    
    # Large Series
    (pd.Series(np.random.rand(1000000)), "float64_1d_large"),
    
    # Sparse Series
    (pd.Series(pd.arrays.SparseArray(np.zeros(1000))), "sparse_pd_series"),
    (pd.Series(pd.arrays.SparseArray(np.random.choice([0, 1.0], size=1000, p=[0.99, 0.01]))), "sparse_pd_series_1_percent"),
    
    # Categorical Data
    (pd.Series(pd.Categorical([0, 1, 2, 0, 1, 2])), "categorical_1d"),
    (pd.Series(pd.Categorical([0, 1, 2, 3, 4, 5])), "categorical_1d_more_categories"),
    
    # Integer Data
    (pd.Series([1, 2, 3], dtype=np.int64), "int64_1d"),
    (pd.Series([1, 2, 3], dtype=np.uint8), "uint8_1d"),
]

# ==== TEST CASES FOR pd.DataFrame ====
df_test_cases = [
    # Basic numerical DataFrames
    (pd.DataFrame(np.array([[1.0, 2.0], [3.0, 4.0]]), columns=["A", "B"]), "float64_2d"),
    (pd.DataFrame(np.array([[5, 6, 7], [8, 9, 10]]), columns=["X", "Y", "Z"]), "int64_2d"),
    
    # Mixed data types
    (pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": ["x", "y", "z"]}), "mixed_numeric_string"),
    (pd.DataFrame({"A": [1, 2, 3], "B": [1.1, 2.2, 3.3], "C": ["a", "b", "c"]}), "mixed_int_float_string"),
    
    # Sparse DataFrames
    (pd.DataFrame(pd.arrays.SparseArray(np.zeros((1000,))), columns=["SparseCol"]), "sparse_pd_dataframe"),
    (pd.DataFrame(pd.arrays.SparseArray(np.random.choice([0, 1.0], size=1000, p=[0.99, 0.01]))), "sparse_pd_dataframe_1_percent"),
    
    # Time Series DataFrames
    (pd.DataFrame({"Date": pd.date_range("2022-01-01", periods=10), "Value": np.random.rand(10)}), "time_series_df"),
    (pd.DataFrame({"Date": pd.date_range("2022-01-01", periods=10, freq="D"), "Category": list("AAABBBCCDD"), "Value": np.random.rand(10)}), "time_series_categorical_df"),
    
    # Large DataFrames
    (pd.DataFrame(np.random.rand(1000, 1000)), "float64_2d_large"),
    
    # Multi-Index DataFrame
    (pd.DataFrame(
        np.random.rand(6, 3),
        index=pd.MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 1), ("B", 2), ("C", 1), ("C", 2)], names=["Group", "Subgroup"]),
        columns=["X", "Y", "Z"]
    ), "multi_index_df"),
]

# Run tests for Series
for series, name in series_test_cases:
    try:
        dctx.from_pandas(series, shared_memory=False).print().compute()
    except Exception as e:
        print(f"Error for {name} (Series): {e}")

# Run tests for DataFrames
for df, name in df_test_cases:
    try:
        dctx.from_pandas(df, shared_memory=False).print().compute()
    except Exception as e:
        print(f"Error for {name} (DataFrame): {e}")
