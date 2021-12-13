#!/usr/bin/python3

import sys
from random import choice


if (len(sys.argv) != 5):
    print("Usage: ./COO_to_Dense.py cooFile.txt numberOfVertices nonZeroValScaleFactor outputFilename")
    exit()

filename = sys.argv[1]
size = int(sys.argv[2])
nonZeroValScaleFactor = int(sys.argv[3])
outputFile = sys.argv[4]


csrMat = [ [] for i in range(size)]
numNonZeros = 0
# Parse CSR
with open(filename) as f:
    for line in f:   
        # Skip comments in csv 
        if line.startswith("#"):
            continue 
        splitted = line.split()
        r = int(splitted[0])
        c = int(splitted[1])
        if c not in csrMat[r]:
            numNonZeros += 1
            csrMat[r].append(c)
        # For symmetry (undirected graph).
        if r not in csrMat[c]:
            numNonZeros += 1
            csrMat[c].append(r)

        
# Increase nonZeroValues
newCsrMat = [ [] for i in range(size)]

# Available edges from csr in reversed order
availableEdges = [edg for edg in range(size -1, -1, -1)]    
addedValues = 0
newCsrMat = csrMat.copy()
# For each row 
for row in range(0, int(len(csrMat))): 
    # Limit of nonZeros to add
    addingLimit = nonZeroValScaleFactor * numNonZeros 

    # Extend row by nonZeroValScaleFactor
    startingLength = len(csrMat[row])
    endLength = len(newCsrMat[row]) * nonZeroValScaleFactor


    for j in range(startingLength, endLength):
        randEdg = choice(availableEdges)
        # If already exists, keep searching
        while randEdg == row or randEdg in newCsrMat[row] or row in newCsrMat[randEdg]:
            randEdg = choice(availableEdges)

        newCsrMat[row].append(randEdg)
        newCsrMat[randEdg].append(row)        
        addedValues+=2

        # If limit reached, remove newly added values until we reach desired number of added values
        if addingLimit <= addedValues + numNonZeros:
            
            while(addingLimit < addedValues + numNonZeros):
                newCsrMat[row].pop()
                newCsrMat[randEdg].pop()
                addedValues -= 2
            break # Exit inner loop
    else:
        # availableEdges has size -> 0 order. Remove last element from available edges and continue
        # we already added values to this now, no need to add more edges here
        availableEdges.pop() 
        continue
    break # Exit outter loop


numNonZeros += addedValues

resFile = open(outputFile, 'w')
oldFile = open(filename, 'r')

# Copy first commented lines
# TODO  fix last row that contains Nodes: numNodes Edges: numEdges,
#       to show correct number of edges
for line in oldFile:
    if line.startswith("#"):
        resFile.write(line)
    else:
        break

oldFile.close()

for row, csrRow in enumerate(newCsrMat):
    for column in csrRow:
        resFile.write(str(row) + "\t" + str(column) + "\n")
resFile.close()
