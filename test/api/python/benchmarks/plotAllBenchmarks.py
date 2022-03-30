from asyncio.subprocess import PIPE
import glob, os
from re import sub
import subprocess
import matplotlib.pyplot as plt
from api.python.utils.consts import PROTOTYPE_PATH
import seaborn as sns
import pandas as pd

daphneset = pd.read_csv("daphneset.csv")
kmeans = pd.read_csv("kmeans.csv")
sumdataset = pd.read_csv("sumdataset.csv")
sns.set_theme(style="whitegrid")
g=sns.catplot(data=sumdataset, kind="bar", x="size", y="time", hue="name", ci="sd", palette="dark", aspect=2).despine(left=True).set(ylim=(0,1000))
ax = g.axes[0,0]
for c in ax.containers:
    ax.bar_label(c)
plt.tight_layout()
g.savefig("/home/dzc/Desktop/out.png")

plt.clf()



sns.set_theme(style="whitegrid")
bp=sns.barplot(x=kmeans["kmeans_name"], y=kmeans["kmeans_runtime"])
bp.bar_label(bp.containers[0])
bp.get_figure().savefig("/home/dzc/Desktop/out2.png")