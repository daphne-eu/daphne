#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results.csv", skipinitialspace=True)

fig, axes = plt.subplots(nrows=2, ncols=2)

df['precompiled_baseline'] = df['precompiled'] / df['precompiled']
df['codegen_speedup'] = df['precompiled'] / df['codegen']
df['codegen_optimized_speedup'] = df['precompiled'] / df['codegen_optimized']

# 10x10 - 100kb
#  df.iloc[0:4].plot(ax=axes[0,0], kind='barh', x='test_name', y=[1,2,3])
#  df.iloc[10:14].plot(ax=axes[0,1], kind='barh', x='test_name', y=[1,2,3])
#  df.iloc[0:4].plot(ax=axes[1,0], kind='line', x='test_name', y=[4,5,6])
#  df.iloc[10:14].plot(ax=axes[1,1], kind='line', x='test_name', y=[4,5,6])

# 1mb - 10mb
#  df.iloc[4:7].plot(ax=axes[0,0], kind='barh', x='test_name', y=[1,2,3])
#  df.iloc[14:17].plot(ax=axes[0,1], kind='barh', x='test_name', y=[1,2,3])
#  df.iloc[4:7].plot(ax=axes[1,0], kind='line', x='test_name', y=[4,5,6])
#  df.iloc[14:17].plot(ax=axes[1,1], kind='line', x='test_name', y=[4,5,6])

# 1gb - 10gb
#  df.iloc[7:10].plot(ax=axes[0,0], kind='barh', x='test_name', y=[1,2,3])
#  df.iloc[17:20].plot(ax=axes[0,1], kind='barh', x='test_name', y=[1,2,3])
#  df.iloc[7:10].plot(ax=axes[1,0], kind='line', x='test_name', y=[4,5,6])
#  df.iloc[17:20].plot(ax=axes[1,1], kind='line', x='test_name', y=[4,5,6])

# all tests
df.iloc[0:10].plot(ax=axes[0,0], kind='barh', x='test_name', y=[1,2,3])
df.iloc[10:20].plot(ax=axes[0,1], kind='barh', x='test_name', y=[1,2,3])
df.iloc[0:10].plot(ax=axes[1,0], kind='line', x='test_name', y=[4,5,6])
df.iloc[10:20].plot(ax=axes[1,1], kind='line', x='test_name', y=[4,5,6])

# precompiled at top
axes[0,0].invert_yaxis()
axes[0,1].invert_yaxis()

axes[0,0].set_title('Float32 Runtime')
axes[0,1].set_title('Float64 Runtime')
axes[1,0].set_title('Float32 Speedup')
axes[1,1].set_title('Float64 Speedup')



#  axes[0, 0].set_xlabel('ms')
#  axes[0, 1].set_xlabel('ms')
axes[0, 0].set_xlabel('ms - log scale')
axes[0, 1].set_xlabel('ms - log scale')

axes[1, 0].set_ylabel('speedup')
axes[1, 1].set_ylabel('speedup')

axes[0,0].set_xscale("log")
axes[0,1].set_xscale("log")
fig.suptitle('i5-8250U\n4 cores 8 threads max 3.4 GHz\n AVX, AVX2, FMA3\nL1 32KB L2 256KB L3 6MB\nPeak: 435.2 GFLOP/s')
plt.show()

## codegen_optimized enables loop vectorize, slpvectorize and unroll loops
