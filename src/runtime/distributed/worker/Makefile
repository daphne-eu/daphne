#this Makefile is only for convenient MPI module testing
#need to be integrated with DAPHNE CMakefiles

CFLAGS = -Wall -ansi -pedantic 
LFLAGS = -lstdc++

MPIProcess: mainMPI.cpp MPIProcess.h
	mpicc $(CFLAGS) mainMPI.cpp -o MPIProcess $(LFLAGS)
# MPIProcess: MPIScatter.cpp
# 	mpicc $(CFLAGS) MPIScatter.cpp -o MPIScatter $(LFLAGS)
clean:
	rm -f MPIProcess *# *~ *.o PI*