#!/usr/bin/env python3

import pandas as pd

import os
import sys
import json

# *****************************************************************************
# Utilities
# *****************************************************************************

def getDict(col):
    return {val: idx for idx, val in enumerate(sorted(col.unique()))}

def dictEncode(col):
    d = getDict(col)
    return col.map(d), d
    
#def writeFileAndMetaData(df, filename):
#    df.to_csv(filename, sep=",", header=False)
#    with open(filename + ".meta", "w") as f):
#        f.write(",".join([len(df), len(df.columns), ]))
        
# *****************************************************************************
# Main
# *****************************************************************************

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------------
    
    if(len(sys.argv) != 2):
        print("Usage: python3 {} <pathData>".format(sys.argv[0]))
        sys.exit(1)
        
    pathData = sys.argv[1]

    # -------------------------------------------------------------------------
    # Preprocess CUSTOMER table
    # -------------------------------------------------------------------------
    
    dfC = pd.read_csv(os.path.join(pathData, "customer.tbl"), sep="|", usecols=range(8), header=None)
    #dfC = dfC.head()
    dfC.columns = [
        "C_CUSTKEY",
        "C_NAME",
        "C_ADDRESS",
        "C_NATIONKEY",
        "C_PHONE",
        "C_ACCTBAL",
        "C_MKTSEGMENT",
        "C_COMMENT"
    ]

    dfc_types =["si64", "si64", "si64", "si64", "si64", "f64", "si64", "si64"]

    #print(dfC)

    for colName in ["C_NAME", "C_ADDRESS", "C_PHONE"]:
        dfC[colName], _ = dictEncode(dfC[colName])
    for colName in ["C_COMMENT"]:
        # Real dictionary encoding is unnecessary and too expensive here.
        dfC[colName] = range(len(dfC))
        
    dfC["C_MKTSEGMENT"], dictMktSegment = dictEncode(dfC["C_MKTSEGMENT"])
    dfC_meta = {}
    
    print("Code for C_MKTSEGMENT == 'AUTOMOBILE': {}".format(dictMktSegment["AUTOMOBILE"]))
    for colName in ["C_NATIONKEY", "C_MKTSEGMENT"]:
        print("{} is between {} and {} ({} distinct values)".format(
                colName,
                dfC[colName].min(),
                dfC[colName].max(),
                len(dfC[colName].unique())
        ))

    #print(dfC)

    dfC.to_csv(os.path.join(pathData, "customer.csv"), sep=",", header=False, index=False)


    dfC_meta["numRows"] = len(dfC)
    dfC_meta["numCols"] = len(dfC.columns)
    dfC_meta["schema"] = []

    for label, type in zip(dfC.columns, dfc_types):
        dfC_meta["schema"].append({"label": label, "valueType": type})


    with open(os.path.join(pathData, "customer.csv.meta"), "w") as f:
        f.write(json.dumps(dfC_meta, indent=4))
    
    # -------------------------------------------------------------------------
    # Preprocess ORDERS table
    # -------------------------------------------------------------------------
    
    dfO = pd.read_csv(os.path.join(pathData, "orders.tbl"), sep="|", usecols=range(9), header=None)
    #dfO = dfO.head()
    dfO.columns = [
        "O_ORDERKEY",
        "O_CUSTKEY",
        "O_ORDERSTATUS",
        "O_TOTALPRICE",
        "O_ORDERDATE",
        "O_ORDERPRIORITY",
        "O_CLERK",
        "O_SHIPPRIORITY",
        "O_COMMENT"
    ]

    #print(dfO)

    for colName in ["O_ORDERSTATUS", "O_ORDERPRIORITY", "O_CLERK"]:
        dfO[colName], _ = dictEncode(dfO[colName])
    for colName in ["O_ORDERDATE"]:
        dfO[colName] = dfO[colName].apply(lambda val: val.replace("-", ""))
    for colName in ["O_COMMENT"]:
        # Real dictionary encoding is unnecessary and too expensive here.
        dfO[colName] = range(len(dfO))
    
    
    print("O_ORDERDATE is between {} and {}".format(dfO["O_ORDERDATE"].min(), dfO["O_ORDERDATE"].max()))
    for colName in ["O_ORDERSTATUS", "O_ORDERPRIORITY", "O_CLERK"]:
        print("{} is between {} and {} ({} distinct values)".format(
                colName,
                dfO[colName].min(),
                dfO[colName].max(),
                len(dfO[colName].unique())
        ))

    #print(dfO)


    dfO_meta = {}
    dfO_meta["numRows"] = len(dfO)
    dfO_meta["numCols"] = len(dfO.columns)
    dfO_meta["schema"] = []

    dfo_types = ["si64", "si64", "si64", "f64", "si64", "si64", "si64", "si64", "si64"]

    for label, type in zip(dfO.columns, dfo_types):
        dfO_meta["schema"].append({"label": label, "valueType": type})

    dfO.to_csv(os.path.join(pathData, "orders.csv"), sep=",", header=False, index=False)
    with open(os.path.join(pathData, "orders.csv.meta"), "w") as f:
        f.write(json.dumps(dfO_meta, indent=4))

