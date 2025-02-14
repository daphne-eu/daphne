<!--
Copyright 2021 The DAPHNE Consortium

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Using external DBMS in DaphneDSL

At the time of writing, Daphne supports two DBMS: DuckDB and SQLite. In addition, there is the option to use DuckDB through ODBC.

To use these features, we can call the `externalSql()` function as follows:

## DuckDB

DuckDB is locally embedded in Daphne, and to use it we have to set the `dbms` to "duckdb". There many ways to use it, including:

### Creating a new .db file

If we do not have a file, we can create one using 

```cpp
externalSql("CREATE TABLE IF NOT EXISTS table_name (id INTEGER, name VARCHAR);", "duckdb", "nameOfMyFile.db");
```

This will create a new file called `nameOfMyFile.db` which we can access later and run queries on it just by putting it as our `connection` parameter.

DuckDB could also be called in memory to execute CSV files, for example:

```cpp
externalSql("SELECT * FROM read_csv_auto('path/to/csvFile.csv')", "duckdb", ":memory:");
```

This will access the `csvFile.csv` file and run the query on it, returning the result as a frame. 

When the connection string is empty DuckDB Defaults to In-Memory Mode.

## ODBC

ODBC is implemented in our kernel to work with any DBMS that supports ODBC. 

The ODBC part of the code is invoked when the `connection` parameter is set to "odbc". In that code a DSN is called with what was passed as `dbms` parameter. 

To use ODBC we must first set up a DSN for the DBMS that we want to use. This is done as follows:

1. Open the `/etc/odbcinst.ini` file  with the command:

```ubuntu
sudo nano /etc/odbcinst.ini
```

and write this inside: 

```ini
[DBMS Driver]
Driver = /path/to/dbms_odbc_driver.so
```

replacing DBMS with the desired one and the path with our path to the DBMS ODBC driver (which needs to be installed).

2. Open the `/etc/odbc.ini` file  with the command:

```ubuntu
sudo nano /etc/odbc.ini
```

and write this inside: 

```ini
[dbms]
Driver = DBMS Driver
Database = /path/to/dataFile.db
allow_unsigned_extensions = true
```

replacing dbms with the desired one, DBMS Driver with the name of the driver as written in the `odbcinst.ini` file and the path to our desired data file in Database. 

If the query is happening in memory, then we can write `:memory:` as Database. 

With that we have set up a DSN with the name dbms. Now to use ODBC we have to call it like this:

```cpp
externalSql("SELECT * FROM table", "dbms", "odbc");
```

This would call ODBC with the DSN `dbms`.

## SQLite


## Limitations

1. The ODBC libraries are not included in the used container. This means that in order to use/compile daphne, we have to download the libraries after connecting to the container like this: 

```ubuntu
sudo apt update
sudo apt install -y unixodbc unixodbc-dev
```

2. Everytime we enter the container, we have to setup the `/etc/odbc.ini` and `/etc/odbcinst.ini` files.