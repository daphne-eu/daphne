from asyncio.subprocess import PIPE
import glob, os
from re import sub
import statistics
import subprocess
from api.python.utils.consts import PROTOTYPE_PATH
import pandas as pd
x = []
y = []
z=[]
yapp = []
ykmn = []
kmeans_name = []
kmeans_runtime = []
for file in glob.glob("*.py"):
    yapp.clear()
    ykmn.clear()
    if "runAllBenchmarks" in file or "plotAllBenchmarks" in file:
           continue
    if "k-means" not in file:
        for i in range(0, 4):
            
            for j in range(0, 10):
                p = subprocess.Popen(["python3", file, str(2**(i*3))], stdout=PIPE)
                yapp.append(float(float(str(p.communicate()[-2]).split("res: 0")[1].replace('\\n', "").replace("'", ""))/10**6))
            x.append(str(2**(i*3))+"x"+str(2**(i*4)))
            y.append(statistics.mean(yapp))
            z.append(file)
        print("Benchmarking complete - filename: "+file)
    else:
            for j in range(0, 10):
                p = subprocess.Popen(["python3", file], stdout=PIPE)
                ykmn.append(float(float(str(p.communicate()[-2]).split("res: 0")[1].replace('\\n', "").replace("'", ""))/10**6))
            kmeans_runtime.append(statistics.mean(ykmn))
            kmeans_name.append(file)
            print("Benchmarking complete - filename: "+file)
      
sumdataset = pd.DataFrame({
    "size":x,
    "time":y,
    "name": z,
})
daphne_progs = []
daphne_results = []
os.chdir(PROTOTYPE_PATH)
res = [f for f in glob.glob("*.daphne") if "bm" in f]
for prog in res:
    for i in range(0, 10):
        yapp = 0
        p = subprocess.Popen(["build/bin/daphne", prog], stdout=PIPE)
        yapp+= (float(str(p.communicate()).split("Time input read: ")[1].split('ms')[0]))
    if "kmeans" not in prog:
            daphne_progs.append(prog)
            daphne_results.append(yapp/10)
    else:
            kmeans_runtime.append(yapp/10)
            kmeans_name.append(prog)
    print("Benchmarking complete - filename: "+prog)
      
daphneset = pd.DataFrame({
    "daphne_progs":daphne_progs,
    "daphne_results":daphne_results})

kmeans = pd.DataFrame({
    "kmeans_name":kmeans_name,
    "kmeans_runtime":kmeans_runtime})

sumdataset.to_csv("test/api/python/benchmarks/sumdataset.csv")
daphneset.to_csv("test/api/python/benchmarks/daphneset.csv")
kmeans.to_csv("test/api/python/benchmarks/kmeans.csv")