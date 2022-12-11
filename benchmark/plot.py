#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results.csv", skipinitialspace=True)
df[df.columns[1:]] = df[df.columns[1:]] / 1e6



ax = df.iloc[0:3].plot(colormap='viridis', kind='barh', x='test_name', y=[3,4,5])
ax.invert_yaxis()

plt.xlabel("runtime in ms")
#  fig = plt.figure()
#  fig.add_subplot(2, 2, 1)   #top and bottom left
#  fig.add_subplot(2, 2, 2)   #top right
#  fig.add_subplot(2, 2, 4)   #bottom right
#  plt.show()

plt.show()

ax = df.iloc[3:6].plot(colormap='viridis', kind='barh', x='test_name', y=[3,4,5])
ax.invert_yaxis()
plt.xlabel("runtime in ms")
plt.show()
