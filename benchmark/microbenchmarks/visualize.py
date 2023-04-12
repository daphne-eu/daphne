#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def niceSize(n):
    if n >= 1000000:
        return "{:.0f}M".format(n / 1000000)
    if n >= 1000:
        return "{:.0f}K".format(n / 1000)
    return n
    
def visualizeLm(prefix):
    dfAll = pd.read_csv(prefix + "/runtimes_lm.csv", sep="\t")

    colInputDim = "Input dimensions (rows × cols)"
    dfAll[colInputDim] = dfAll.apply(lambda row: "{} × {}".format(niceSize(row["numRows"]), niceSize(row["numCols"])), axis=1)
    dfAll["Runtime [s]"] = dfAll["runtime [ns]"] / (1000 * 1000 * 1000)
    dfAll["system"] = dfAll["system"].apply(lambda s: s.replace("TensorFlow", "TF"))
    
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
        
    sns.barplot(
        ax=ax,
        y="Runtime [s]", ci=None, x=colInputDim, hue="system", palette=sns.color_palette("Paired"),
        data=dfAll
    )
    
    ax.legend(
        bbox_to_anchor=(0.01, 1),
        loc="upper left",
        handlelength=1,
        labelspacing=0.25,
        handletextpad=0.4,
        borderpad=0,
        borderaxespad=0,
        frameon=False,
    )
    sns.despine()
    
    fig.savefig(prefix + "/micro_lm.pdf", bbox_inches="tight")
    ax.set_yscale("log", base=10)
    fig.savefig(prefix + "/micro_lm_logy.pdf", bbox_inches="tight")
    
def visualizeKmeans(prefix):
    dfAll = pd.read_csv(prefix + "/runtimes_kmeans.csv", sep="\t")

    colInputDim = "Input dimensions (rows × cols)"
    dfAll[colInputDim] = dfAll.apply(lambda row: "{} × {}".format(niceSize(row["numRecords"]), niceSize(row["numFeatures"])), axis=1)
    dfAll["Runtime [s]"] = dfAll["runtime [ns]"] / (1000 * 1000 * 1000)
    dfAll["system"] = dfAll["system"].apply(lambda s: s.replace("TensorFlow", "TF"))
    
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    
    sns.color_palette("Paired")
    sns.barplot(
        ax=ax,
        y="Runtime [s]", ci=None, x=colInputDim, hue="system", palette=sns.color_palette("Paired"),
        data=dfAll
    )
    
    ax.legend(
        bbox_to_anchor=(0.01, 1),
        loc="upper left",
        handlelength=1,
        labelspacing=0.25,
        handletextpad=0.4,
        borderpad=0,
        borderaxespad=0,
        frameon=False,
    )
    sns.despine()
    
    fig.savefig(prefix + "/micro_kmeans.pdf", bbox_inches="tight")
    ax.set_yscale("log", base=10)
    fig.savefig(prefix + "/micro_kmeans_logy.pdf", bbox_inches="tight")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: {} <INT useLm> <INT useKmeans> <results_path>".format(sys.argv[0]))

    useLm = bool(int(sys.argv[1]))
    useKmeans = bool(int(sys.argv[2]))

    sns.set_context("paper", 1.4)
    
    if useLm:
        visualizeLm(sys.argv[3])
    if useKmeans:
        visualizeKmeans(sys.argv[3])
