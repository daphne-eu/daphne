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
    numRecords   = int(sys.argv[1])
    numFeatures  = int(sys.argv[2])
    numCentroids = int(sys.argv[3])
    numIter      = int(sys.argv[4])
    singleThreaded = bool(int(sys.argv[5]))
    useInputFiles  = bool(int(sys.argv[6]))
    inputFileX = sys.argv[7]
    inputFileC = sys.argv[8]
    
    if singleThreaded:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    if useInputFiles:
        dfX = pd.read_csv(inputFileX, sep=",", header=None)
        dfC = pd.read_csv(inputFileC, sep=",", header=None)
        X = tf.convert_to_tensor(dfX.values, dtype=tf.float64)
        C = tf.convert_to_tensor(dfC.values, dtype=tf.float64)
    else:
        X = tf.random.uniform([numRecords  , numFeatures], minval=0.0, maxval=1.0, dtype=tf.float64)
        C = tf.random.uniform([numCentroids, numFeatures], minval=0.0, maxval=1.0, dtype=tf.float64)
    
    tp0 = datetime.datetime.now()
    
    for i in range(numIter):
        # Compute Euclidean squared distances from records (X rows) to centroids
        # (C rows) without the C-independent term, then take the minimum for each
        # record.
        D = -2.0 * tf.matmul(X, C, transpose_b=True) + tf.transpose(tf.math.reduce_sum(tf.math.pow(C, 2.0), axis=1, keepdims=True))
        minD = tf.math.reduce_min(D, axis=1, keepdims=True);

        # Find the closest centroid for each record
        P = tf.cast(D <= minD, tf.float64)
        # If some records belong to multiple centroids, share them equally
        P = P / tf.math.reduce_sum(P, axis=1, keepdims=True)
        # Compute the column normalization factor for P
        P_denom = tf.math.reduce_sum(P, axis=0, keepdims=True)
        # Compute new centroids as weighted averages over the records
        C = tf.matmul(P, X, transpose_a=True) / tf.transpose(P_denom)
    
    tp1 = datetime.datetime.now()
    print((tp1 - tp0).total_seconds() * 1000 * 1000 * 1000, file=sys.stderr, end="")
    
    print(pd.DataFrame(C.numpy()).to_csv(None, header=None, index=None))
