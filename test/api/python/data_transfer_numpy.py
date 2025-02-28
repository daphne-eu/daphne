import numpy as np
from daphne.context.daphne_context import DaphneContext

dctx = DaphneContext()

test_cases = [
    # 1D arrays
    (np.array([1.0, 2.0, 3.0], dtype=np.float64).reshape(-1, 1), "float64_1d"),
    (np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float64).reshape(-1, 1), "float64_1d_longer"),
    (np.array([8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float64).reshape(-1, 1), "float64_1d_even_longer"),
    
    # 2D arrays
    (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), "float64_2d"),
    (np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float64), "float64_2d_wider"),
    (np.array([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]], dtype=np.float64), "float64_2d_taller"),
    (np.array([[17.0, 18.0, 19.0], [20.0, 21.0, 22.0], [23.0, 24.0, 25.0]], dtype=np.float64), "float64_2d_square"),
    
    # Edge cases
    (np.array([np.nan, np.nan, np.nan], dtype=np.float64).reshape(-1, 1), "float64_nan"),
    (np.array([np.inf, -np.inf, np.inf], dtype=np.float64).reshape(-1, 1), "float64_inf"),
    (np.array([1, 2.0, 3], dtype=np.float64).reshape(-1, 1), "float64_mixed"),
    (np.array([-1.0, -2.0, -3.0], dtype=np.float64).reshape(-1, 1), "float64_negative"),
    (np.array([1e-10, 2e-10, 3e-10], dtype=np.float64).reshape(-1, 1), "float64_small"),
    (np.array([1e10, 2e10, 3e10], dtype=np.float64).reshape(-1, 1), "float64_large"),
    
    # Higher-dimensional arrays
    (np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64), "float64_3d"),
    (np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]], [[17.0, 18.0], [19.0, 20.0]]], dtype=np.float64), "float64_3d_larger"),
    (np.array([[[21.0, 22.0, 23.0], [24.0, 25.0, 26.0]], [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0]]], dtype=np.float64), "float64_3d_wider"),
    (np.array([[[33.0, 34.0], [35.0, 36.0], [37.0, 38.0]], [[39.0, 40.0], [41.0, 42.0], [43.0, 44.0]]], dtype=np.float64), "float64_3d_taller"),
    (np.random.rand(2, 3, 4, 5), "float64_4d"),
    (np.random.rand(2, 2, 3, 4, 5), "float64_5d"),
    
    # Very large arrays
    (np.random.rand(1000000).reshape(-1, 1), "float64_1d_large"),
    (np.random.rand(1000, 1000), "float64_2d_large"),
    
    # Sparse arrays
    (np.zeros((1000, 1000)), "sparse_np_array"),  
    (np.random.choice([0, 1.0], size=(1000, 1000), p=[0.99, 0.01]), "sparse_np_array_1_percent"),  # 1% non-zero
    (np.random.choice([0, 1.0], size=(1000, 1000), p=[0.95, 0.05]), "sparse_np_array_5_percent"),  # 5% non-zero
    (np.random.choice([0, 1.0], size=(1000, 1000), p=[0.9, 0.1]), "sparse_np_array_10_percent"),  # 10% non-zero
    
    # Categorical data (using float representation)
    (np.array([0, 1, 2, 0, 1, 2], dtype=np.float64).reshape(-1, 1), "categorical_1d"),
    (np.array([0, 1, 2, 3, 4, 5], dtype=np.int32).reshape(-1, 1), "categorical_1d_more_categories"),
    (np.array([0, 1, 0, 1, 0, 1], dtype=np.int32).reshape(-1, 1), "categorical_1d_fewer_categories"),
    (np.array([0, 0, 0, 0, 0, 0], dtype=np.int32).reshape(-1, 1), "categorical_1d_single_category"),
    (np.array([0, 1, 2, 1, 0, 2, 1, 0, 2], dtype=np.int32).reshape(-1, 1), "categorical_1d_repeated_categories"),

    # Different data types
    (np.array([1, 2, 3], dtype=np.int64).reshape(-1, 1), "int64_1d"),
    (np.array([1, 2, 3], dtype=np.uint8).reshape(-1, 1), "uint8_1d"),
    
    # Non-standard shapes
    (np.random.rand(1, 1000), "float64_1x1000"),
    (np.random.rand(1000, 1), "float64_1000x1"),
]

for X, name in test_cases:
    try:

        dctx.from_numpy(X, shared_memory=False).print().compute()

    except Exception as e:
        print(f"Error for {name}: {e}")