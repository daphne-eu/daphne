#!/usr/bin/env python3

import datetime
import os
import sys
import pandas as pd
import numpy as np

# TODO Check what messages it actually prints, maybe some of that is important.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

if __name__ == "__main__":
    numRows = int(sys.argv[1])
    numCols = int(sys.argv[2])
    singleThreaded = bool(int(sys.argv[3]))
    useInputFile = bool(int(sys.argv[4]))
    inputFile = sys.argv[5]
    
    if singleThreaded:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    if useInputFile:
        df = pd.read_csv(inputFile, sep=",", header=None)
        XY = tf.convert_to_tensor(df.values, dtype=tf.float64)
    else:
        XY = tf.random.uniform([numRows, numCols], minval=0.0, maxval=1.0, dtype=tf.float64)
    
    tp0 = datetime.datetime.now()
    
    X = tf.gather(XY, range(numCols - 1), axis=1)
    y = tf.gather(XY, [numCols - 1], axis=1)
    
    # -------------------------------------------------------------------------
    # Normalize and standardize columns
    # -------------------------------------------------------------------------
    
    X = (X - tf.math.reduce_mean(X, 0)) / tf.math.reduce_std(X, 0);

    # -------------------------------------------------------------------------
    # Linear regression
    # -------------------------------------------------------------------------
    
    X = tf.concat([X, np.ones((numRows, 1))], 1)
  
    lamda = tf.constant(
        0.001 * np.ones((numCols, 1), dtype=float),
        dtype=tf.float64,
        shape=[numCols]
    )
    
    A = tf.matmul(X, X, transpose_a=True)
    b = tf.matmul(X, y, transpose_a=True)
    A = A + tf.linalg.diag(lamda)
    beta = tf.linalg.solve(A, b)
    
    tp1 = datetime.datetime.now()
    print((tp1 - tp0).total_seconds() * 1000 * 1000 * 1000, file=sys.stderr, end="")
    
    for val in beta.numpy():
        print(val[0])
