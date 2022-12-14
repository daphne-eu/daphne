#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("speedup.csv", skipinitialspace=True)

#  fig, axes = plt.subplots(nrows=1, ncols=1)

print(df)
# all tests
df.plot(title='Sum Reduce with F32 cast to F64. f32_10gb precompiled did not'
        'finish with 24GB ram!',kind='bar', x='test_name', y=[1,2,3, 4, 5])
#  df.iloc[0:10].plot(ax=axes[1], kind='line', x='test_name', y=[4,5,6])

# precompiled at top
#  axes[0].invert_yaxis()
#  axes[0].set_xlabel('ms - log scale')

#  axes[1].set_ylabel('speedup')

#  axes[0].set_xscale("log")

#  fig.suptitle('i5-8250U\n4 cores 8 threads max 3.4 GHz\n AVX, AVX2, FMA3\nL1 32KB L2 256KB L3 6MB\nPeak: 435.2 GFLOP/s')
#  fig.suptitle('Delta - AMD EPYC 7302 16-Core\n')
plt.show()

## codegen_optimized enables loop vectorize, slpvectorize and unroll loops
