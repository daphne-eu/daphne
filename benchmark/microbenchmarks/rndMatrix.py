#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

if __name__ == "__main__":
    numRows = int(sys.argv[1])
    numCols = int(sys.argv[2])
    outFilePath = sys.argv[3]

    XY = tf.random.uniform([numRows, numCols], minval=0.0, maxval=1.0, dtype=tf.float64)
    
    pd.DataFrame(XY.numpy()).to_csv(outFilePath, sep=",", header=None, index=None)
    
    with open(outFilePath + ".meta", "w") as outFile:
        outFile.write("{},{},1,f64".format(numRows, numCols))
