# StarSchemaBenchmark
[![Build Status](https://travis-ci.org/lemire/StarSchemaBenchmark.png)](https://travis-ci.org/lemire/StarSchemaBenchmark)

This is a simple copy of the table generation code from O'Neil et al.'s Star Schema Benchmark as
described in the following paper:

Patrick O'Neil, Elizabeth (Betty) O'Neil and Xuedong Chen. "The Star Schema Benchmark," Online Publication of Database Generation program., January 2007.
http://www.cs.umb.edu/~poneil/StarSchemaB.pdf


# Usage

        make
        
        (customer.tbl)
        dbgen -s 1 -T c
        
        (part.tbl)
        dbgen -s 1 -T p
        
        (supplier.tbl)
        dbgen -s 1 -T s
        
        (date.tbl)
        dbgen -s 1 -T d
        
        (fact table lineorder.tbl)
        dbgen -s 1 -T l
        
        (for all SSBM tables)
        dbgen -s 1 -T a
        
These commands should generate the following files: customer.tbl  date.tbl      lineorder.tbl part.tbl      supplier.tbl

You can easily generate larger files by modifying the scale parameter (-s).


To generate the refresh (insert/delete) data set: 

        dbgen -s 1 -r 5 -U 4

where "-r 5" specifies refreshin fact n/10000 "-U 4" specifies 4 segments for deletes and inserts. This will create the files create delete.[1-4] and lineorder.tbl.u[1-4] with refreshing fact 0.05%.


# Suitability

This is strictly for research purposes. 
