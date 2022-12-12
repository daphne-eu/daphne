#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results.csv", skipinitialspace=True)
df[df.columns[1:]] = df[df.columns[1:]] / 1e6


fig, axes = plt.subplots(nrows=2, ncols=2)

df['precompiled_baseline'] = df['precompiled'] / df['precompiled']
df['codegen_speedup'] = df['precompiled'] / df['codegen']
df['codegen_optimized_speedup'] = df['precompiled'] / df['codegen_optimized']

# 10x10 - 100kb
#  df.iloc[0:4].plot(ax=axes[0,0], kind='barh', x='test_name', y=[3,4,5])
#  df.iloc[10:14].plot(ax=axes[0,1], kind='barh', x='test_name', y=[3,4,5])
#  df.iloc[0:4].plot(ax=axes[1,0], kind='line', x='test_name', y=[6, 7, 8])
#  df.iloc[10:14].plot(ax=axes[1,1], kind='line', x='test_name', y=[6, 7, 8])

# 1mb - 10mb
#  df.iloc[4:7].plot(ax=axes[0,0], kind='barh', x='test_name', y=[3,4,5])
#  df.iloc[14:17].plot(ax=axes[0,1], kind='barh', x='test_name', y=[3,4,5])
#  df.iloc[4:7].plot(ax=axes[1,0], kind='line', x='test_name', y=[6, 7, 8])
#  df.iloc[14:17].plot(ax=axes[1,1], kind='line', x='test_name', y=[6, 7, 8])

# 1gb - 10gb
#  df.iloc[7:10].plot(ax=axes[0,0], kind='barh', x='test_name', y=[3,4,5])
#  df.iloc[17:20].plot(ax=axes[0,1], kind='barh', x='test_name', y=[3,4,5])
#  df.iloc[7:10].plot(ax=axes[1,0], kind='line', x='test_name', y=[6, 7, 8])
#  df.iloc[17:20].plot(ax=axes[1,1], kind='line', x='test_name', y=[6, 7, 8])

# all tests
df.iloc[0:10].plot(ax=axes[0,0], kind='barh', x='test_name', y=[3,4,5])
df.iloc[10:20].plot(ax=axes[0,1], kind='barh', x='test_name', y=[3,4,5])
df.iloc[0:10].plot(ax=axes[1,0], kind='line', x='test_name', y=[6, 7, 8])
df.iloc[10:20].plot(ax=axes[1,1], kind='line', x='test_name', y=[6, 7, 8])

# precompiled at top
axes[0,0].invert_yaxis()
axes[0,1].invert_yaxis()

axes[0, 0].set_xlabel('runtime in ms - log scale')
axes[0, 1].set_xlabel('runtime in ms - log scale')

axes[1, 0].set_ylabel('speedup')
axes[1, 1].set_ylabel('speedup')

axes[0,0].set_xscale("log")
axes[0,1].set_xscale("log")
plt.show()

## codegen_optimized enables loop vectorize, slpvectorize and unroll loops
