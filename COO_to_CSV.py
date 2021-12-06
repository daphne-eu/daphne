#!/usr/bin/python3

import sys
import numpy as np

if (len(sys.argv) == 2 and sys.argv[1] == "--help"):
    print("Usage: ./COO_to_Dense.py cooFile.txt numberOfVertices numberOfWorkers outputFilename outputFormat")
    print("")
    print("Specify worker address list inside the script.")
    exit()

if (len(sys.argv) != 6):
    print("Usage: ./COO_to_Dense.py cooFile.txt numberOfVertices numberOfWorkers outputFilename outputFormat")
    exit()

filename = sys.argv[1]
size = int(sys.argv[2])
numWorkers = int(sys.argv[3])
outputFile = sys.argv[4]
outputFormat = sys.argv[5]


# Worker addresses
workerAddressList = [ "localhost:" + str(i) for i in range(50000, 50000 + numWorkers)]

if (numWorkers > len(workerAddressList)):    
    print("You must specify addresses for all workers (inside the script).")
    exit()


availableFormats = ["DenseMatrix", "COOFormat"]
if (outputFormat not in availableFormats):
    print("Available matrix representation support: ")
    print(availableFormats)
    exit()

######################### DO NOT EDIT #########################

# Split among workers
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


handlesFile = open(outputFile + "_handles.csv", 'w')

# Metadata file
handlesFileMeta = open(outputFile + "_handles.csv.meta", 'w')
handlesFileMeta.write(str(size) + "," + str(size) + ",1,f64")
handlesFileMeta.close()


if (outputFormat == "DenseMatrix"):
    mat = np.zeros(shape=(size, size))
    
    # Parse Dense
    with open(filename) as f:
        for line in f:
            # Skip comments in csv 
            if line.startswith("#"):
                continue 
            splitted = line.split()
            r = int(splitted[0])
            c = int(splitted[1])
            mat[r-1][c-1] = 1
            mat[c-1][r-1] = 1
    
    # Store matrix (optional)
    # foriginalMat = open(outputFile + ".csv", 'w')
    # foriginalMatMeta = open(outputFile + ".csv.meta", 'w')
    # foriginalMatMeta.write(str(size) + "," + str(size) + ",1,f64")

    # for r in mat:
    #     for i,c in enumerate(r):
    #         foriginalMat.write(str(int(c)) + ".0")
    #         if (i == len(r) - 1):
    #             break
    #         foriginalMat.write(",")
    #     foriginalMat.write("\n")

    # Create segments, this functions works exactly like Distribute.h Kernel
    rowSegments = list(split(list(range(size)), numWorkers))    

    for i, rowSegment in enumerate(rowSegments):        
        outputFilename = outputFile + "_" + str(i) + ".csv"

        fcsv = open(outputFilename, 'w')
        
        fcsvMeta = open(outputFilename + ".meta", 'w')
        fcsvMeta.write(str(len(rowSegment)) + "," + str(size) + ",1,f64")
        fcsvMeta.close()

        for row in rowSegment:
            r = mat[row]
            for j,c in enumerate(r):
                fcsv.write(str(int(c)) + ".0")
                if (j == len(r) - 1):
                    break
                fcsv.write(",")
            fcsv.write("\n")

        # Handles file (master)
        # [address, filename, DistributedIndexRow, DistributedIndexCol, numRows, numCols]        
        handlesFile.write(workerAddressList[i] + "," + outputFilename + "," + str(i) + ",0," + str(len(rowSegment)) + "," + str(size) + "\n")


if (outputFormat == "COOFormat"):

    csrMat = [ [] for i in range(size)]
    # Parse CSR
    with open(filename) as f:
        for line in f:   
            # Skip comments in csv 
            if line.startswith("#"):
                continue 
            splitted = line.split()
            r = int(splitted[0])
            c = int(splitted[1])
            csrMat[r].append(c)
            # For symmetry. If statement is needed for duplicates, because some edges present symmetry, but some don't
            if r not in csrMat[c]:
                csrMat[c].append(r)
    
    # Create segments, this functions works exactly like Distribute.h Kernel
    rowSegments = list(split(list(range(size)), numWorkers))

    for i, rowSegment in enumerate(rowSegments):        
        outputFilename = outputFile + "_" + str(i) + ".csv"

        fcsv = open(outputFilename, 'w')
        
        fcsvMeta = open(outputFilename + ".meta", 'w')
        fcsvMeta.write(str(len(rowSegment)) + "," + str(size) + ",1,f64")
        fcsvMeta.close()

        for row in rowSegment:
            connectedVertices = csrMat[row]
            for vertex in connectedVertices:
                fcsv.write(str(row) + "," + str(vertex) + "\n")

        # Handles file (master)
        # [address, filename, DistributedIndexRow, DistributedIndexCol, numRows, numCols]        
        handlesFile.write(workerAddressList[i] + "," + outputFilename + "," + str(i) + ",0," + str(len(rowSegment)) + "," + str(size) + "\n")
