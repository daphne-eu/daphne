#!/usr/bin/env python3

#*********************************************************************************************
# Copyright (C) 2019 by MorphStore-Team                                                      *
#                                                                                            *
# This file is part of MorphStore - a compression aware vectorized column store.             *
#                                                                                            *
# This program is free software: you can redistribute it and/or modify it under the          *
# terms of the GNU General Public License as published by the Free Software Foundation,      *
# either version 3 of the License, or (at your option) any later version.                    *
#                                                                                            *
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
# See the GNU General Public License for more details.                                       *
#                                                                                            *
# You should have received a copy of the GNU General Public License along with this program. *
# If not, see <http://www.gnu.org/licenses/>.                                                *
#*********************************************************************************************

"""
Order-preserving dictionary coding for CSV files.

An input CSV file must fulfil the following criteria:
- The file extension is ".tbl".
- There is no header line, only data.
- Columns are separated by the pipe character "|".
- Lines are separated by the newline character "\\n".
- Each row has a trailing column separator.
This is the format used, e.g., by the Star Schema Benchmark's dbgen tool.

An order-preserving dictionary coding is applied to each string-valued column
of such a CSV file. More precisely, each column is encoded individually, i.e.,
with its own dictionary. Dictionary codes start with zero.

For each input CSV file, the output consists of the following:
- A CSV file projected to the required columns. All columns in this file
  contain only integers, i.e., the values in non-integer columns of the input
  CSV-file are replaced by dictionary codes from the respective column's
  dictionary. These files are required for loading data into a DBMS such as
  MonetDB. They are stored in the sub-directory "tbls_dict".
- For each required string-valued column: A plain text file containing the dictionary of
  this column. In this file, the i-th line contains the value whose dictionary
  code is i (line counting starts with zero). These files are required for
  replacing string constants in queries (see MorphStore's qdict.py tool). They
  are stored in the sub-directory "dicts".
- For each required column: A binary file containing some meta data and the column's
  data elements as an array of uncompressed 64-bit integers in little-endian
  byte ordering. This is exactly the file format of MorphStore's binary_io
  persistence class. These files are required for loading data into MorphStore.
  They are stored in the sub-directory "cols_dict".
  
Furthermore, the names of the columns must be known in order to name the column
files appropriately. Therefore, schema information must be provided as a JSON
file fulfilling the following criteria:
- It contains a single object.
- For each table in the schema, this object must have an attribute named after
  the table.
- The value of this attribute must be a list of the names of the columns of
  this table in the same order as they occur in the CSV file.
For instance, the contents of the schema file could look like this:

{
    "table1": ["col11", "col12"],
    "table2": ["col21", "col22", "col23"]
}

Moreover, a second schema file must be provided, which specifies the required
columns, i.e., the columns to actually take into account. This is motivated by
the fact that benchmark queries might not access all columns of the schema. In
the simplest case, this second schema file can be the same as the first. Then,
all columns are considered.
  
Known limitations:
- Currently, any type other than unsigned integer will be treated as a string.
  However, for floats, dates, etc. there may be more processing-friendly ways
  to obtain integers from them.
- This script is not tuned for performance.
"""

# TODO A cleverer way to determine which columns shall be dictionary coded.
# TODO Reasonable treatment of columns of type float, date, etc..
# TODO Documentation for parameters and return values.


import argparse
import json
import os
import struct
import sys

            
_COLSEP = "|"


def _isInt(val):
    """Returns True if the given string represents an unsigned integer."""
    
    return all(["0" <= c <= "9" for c in val])
            
def _encodeTable(
    tblName,
    colNamesFull,
    colNamesRequired,
    inTblFilePath,
    outTblFilePath,
    outDictDirPath,
    outColDirPath,
    outStatFilePath
):
    """
    Applies order-preserving dictionary coding to all required non-integer
    columns of the given CSV file and creates all output files for this CSV
    file.
    """
    
    print("Processing table '{}'".format(tblName))
    
    countColsFull = len(colNamesFull)
    countColsRequired = len(colNamesRequired)
    print("\t{}/{} columns required.".format(countColsRequired, countColsFull))
    
    # Find out the positions of the columns to be taken into account.
    colIdxsRequired = [
        colIdx
        for colIdx, colName in enumerate(colNamesFull)
        if colName in colNamesRequired
    ]
    
    with open(inTblFilePath, "r") as inTblFile:
        # ---------------------------------------------------------------------
        # Preparation
        # ---------------------------------------------------------------------
        
        # Check if the number of columns is ok.
        firstLine = inTblFile.readline().rstrip()
        firstLineEntries = firstLine.split(_COLSEP)[:-1]
        countColsFound = len(firstLineEntries)
        if countColsFound != countColsFull:
            raise RuntimeError(
                "the number of columns found in the CSV file is {}, but the "
                "number of columns according to the schema file is {}".format(
                    countColsFound, countColsFull
                )
            )
        
        # Determine the columns requiring dictionary coding.
        # TODO This might produce false negatives, since only the first element
        #      of each column is considered.
        nonIntColIdxs = [
            idx
            for idx in colIdxsRequired
            if not _isInt(firstLineEntries[idx])
        ]
        print("\t{}/{} required columns need dictionary coding.".format(
            len(nonIntColIdxs), countColsRequired,
        ))
        
        # ---------------------------------------------------------------------
        # First pass: Create the dictionaries for all non-integer columns
        # ---------------------------------------------------------------------
        
        print("\tCreating dictionaries... ", end="")
        sys.stdout.flush()
        
        inTblFile.seek(0, 0)
        
        countRows = 0;
        
        # Determine the number of rows and the distinct values as a set.
        distValsByColIdx = {idx: set() for idx in nonIntColIdxs}
        if len(nonIntColIdxs):
            for line in inTblFile:
                countRows += 1
                lineEntries = line.rstrip().split(_COLSEP)
                for idx in nonIntColIdxs:
                    distValsByColIdx[idx].add(lineEntries[idx])
        else:
            for line in inTblFile:
                countRows += 1
                    
        # Sort distinct values as a list and write dictionary files.
        for idx in nonIntColIdxs:
            distValsByColIdx[idx] = list(sorted(distValsByColIdx[idx]))
            outDictFilePath = os.path.join(
                outDictDirPath, "{}.{}.dict".format(tblName, colNamesFull[idx])
            )
            with open(outDictFilePath, "w") as outDictFile:
                for val in distValsByColIdx[idx]:
                    outDictFile.write(val + "\n")
                    
        # Create encoding dictionaries as a dict.
        dictByColIdx = {
            # str() because its used with join() later.
            idx: {
                val: str(pos)
                for pos, val in enumerate(distValsByColIdx[idx])
            }
            for idx in nonIntColIdxs
        }
        
        print("done." if len(nonIntColIdxs) else "nothing to do.")
        
        # ---------------------------------------------------------------------
        # Second pass: Encode non-integer columns, create output files
        # ---------------------------------------------------------------------
        
        print("\tEncoding data... ", end="")
        sys.stdout.flush()
        
        inTblFile.seek(0, 0)
        
        # Open output files for all columns.
        outColFiles = {
            idx: open(
                os.path.join(
                    outColDirPath,
                    "{}.{}.uncompr_f.bin".format(tblName, colNamesFull[idx])
                ),
                "wb"
            )
            for idx in colIdxsRequired
        }
        
        # Write metadata for all columns.
        for idx in colIdxsRequired:
            # TODO Does "Q" guarantee 64 bits on all platforms?
            # The number of data elements in the column (logical size).
            outColFiles[idx].write(struct.pack("<Q", countRows))
            # The column's size in byte (physical size).
            outColFiles[idx].write(struct.pack("<Q", countRows * 8))
        
        # TODO Implement this in a cleaner way.
        # ...
        maxVals = {idx: 0 for idx in colIdxsRequired}
        
        # Encode non-integer columns, write output files.
        st = struct.Struct("<Q")
        with open(outTblFilePath, "w") as outTblFile:
            for line in inTblFile:
                lineEntries = line.split(_COLSEP)
                for idx in nonIntColIdxs:
                    lineEntries[idx] = dictByColIdx[idx][lineEntries[idx]]
                outTblFile.write(_COLSEP.join(
                    [lineEntries[idx] for idx in colIdxsRequired]
                ))
                outTblFile.write("\n")
                for idx in colIdxsRequired:
                    entryAsInt = int(lineEntries[idx])
                    outColFiles[idx].write(st.pack(entryAsInt))
                    if entryAsInt > maxVals[idx]:
                        maxVals[idx] = entryAsInt
        
        with open(outStatFilePath, "w") as outStatFile:
            json.dump(
                dict(
                    {colNamesFull[idx]: maxVals[idx] for idx in colIdxsRequired},
                    _cardinality=countRows,
                ),
                outStatFile,
                indent=2
            )
        
        # Close output files for all columns.
        for idx, outColFile in outColFiles.items():
            outColFile.close()
            
        print("done.")
        
def _makeDirIf(dirPath):
    """Creates the specified directory if it does not already exist."""
    
    if os.path.exists(dirPath):
        if os.path.isdir(dirPath):
            return
        else:
            raise RuntimeError(
                "'{}' already exists, but it is not a directory".format(
                    dirPath
                )
            )
    else:
        os.mkdir(dirPath)


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------------
    
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Required positional arguments.
    parser.add_argument(
        "schemaFullFilePath", metavar="schemaFullFile",
        help="The JSON file containing the schema information (all columns)."
        # TODO Validate existence.
    )
    parser.add_argument(
        "schemaRequiredFilePath", metavar="schemaRequiredFile",
        help="The JSON file containing the schema information (required "
            "columns only). Use the same file as for schemaFullFile to take "
            "all columns into account."
        # TODO Validate existence.
    )
    parser.add_argument(
        "inTblDirPath", metavar="inTableDir",
        help="The directory containing the input CSV files. From this "
            "directory, all files with the extension '.tbl' are processed."
        # TODO Validate existence.
    )
    parser.add_argument(
        "outDirPath", metavar="outDir",
        help="The directory where the output sub-directories shall be created "
            "and where the output files shall be stored."
        # TODO Validate existence.
    )
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # Actual work
    # -------------------------------------------------------------------------

    outTblDirPath  = os.path.join(args.outDirPath, "tbls_dict")
    outDictDirPath = os.path.join(args.outDirPath, "dicts")
    outColDirPath  = os.path.join(args.outDirPath, "cols_dict")
    outStatDirPath = os.path.join(args.outDirPath, "stats_dict")
    
    _makeDirIf(outTblDirPath)
    _makeDirIf(outDictDirPath)
    _makeDirIf(outColDirPath)
    _makeDirIf(outStatDirPath)

    with open(args.schemaFullFilePath, "r") as schemaFile:
        schemaFull = json.load(schemaFile)
    with open(args.schemaRequiredFilePath, "r") as schemaFile:
        schemaRequired = json.load(schemaFile)
    
    fileNames = os.listdir(args.inTblDirPath)
    for inTblFileName in fileNames:
        inTblFilePath = os.path.join(args.inTblDirPath, inTblFileName)
        tblName, ext = os.path.splitext(inTblFileName)
        if ext == ".tbl" and os.path.isfile(inTblFilePath):
            _encodeTable(
                tblName,
                schemaFull[tblName],
                schemaRequired[tblName],
                inTblFilePath,
                os.path.join(outTblDirPath, inTblFileName),
                outDictDirPath,
                outColDirPath,
                os.path.join(outStatDirPath, "{}.json".format(tblName))
            )
