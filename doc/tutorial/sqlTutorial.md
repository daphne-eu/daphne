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

# Using SQL in DaphneDSL

DAPHNE supports a rudimentary version of SQL. At any point in a DaphneDSL script, we can execute a SQL query on frames.
We need two operations to achieve this: ```registerView(...)``` and ```sql(...)```

For the following examples we assume we already have a DaphneDSL script which includes calculations on a frame "x" that has the columns "a", "b" and "c".

## General Procedure

### registerView(...)

RegisterView registers a frame for the sql operation.
If we want to execute a SQL query on a frame, we *need* to register it before that.
The operation has two inputs: the name of the table, as a string, and the frame which shall be associated with the given name.

For example, we can register the frame "x", from previous calculations, under the name "Table1". The DaphneDSL script for this would look like this:

```cpp
registerView("Table1", x);
```

### sql(...)

Now that we have registered the tables, that we need for our SQL query, we can go ahead and execute our query. The SQL operation takes one input: the SQL query, as a string. In it, we will reference the table names we previously have registered via registerView(...). As a result of this operation, we get back a frame. The columns of the frame are named after the projection arguments inside the SQL query.

For example, we want to return all the rows of the frame x, which we have previously registered under the name "Table1", where the column "a" is greater than 5 and save it in a new frame named "y". The DaphneDSL script for this would look like this:

```cpp
y = sql("SELECT t.a as a, t.b as b, t.c as c FROM Table1 as t WHERE t.a > 5;");
```

This results in a frame "y" that has three columns "a", "b" and "c".
On the frame y we can continue to build our DaphneDSL script.

## Features

We don't support the complete SQL standard at the moment. For instance, we need to fully specify on which columns we want to operate. In the example above, we see "t.a" instead of simply "a".
Also, not supported are DDL and DCL Queries. Our goal for DML queries is to only support SELECT-statements.
Other features we do and don't support right now can be found below.

### Supported Features

* Cross Product
* Complex Where Clauses
* Inner Join with single and multiple join conditions separated by an "AND" Operator
* Group By Clauses
* Having Clauses
* Order By Clauses
* As

### Not Yet Supported Features

* The Star Operator \*
* Nested SQL Queries like: ```SELECT a FROM x WHERE a IN SELECT a FROM y```
* All Set Operations (Union, Except, Intersect)
* Recursive SQL Queries
* Limit
* Distinct

## Examples

In the following, we show two simple examples of SQL in DaphneDSL.
The DaphneDSL scripts can be found in `doc/tutorial/sqlExample1.daph` and `doc/tutorial/sqlExample2.daph`.

### Example 1

```cpp
//Creation of different matrices for a Frame
    //seq(a, b, c) generates a sequences of the form [a, b] and step size c
    employee_id = seq(1, 20, 1);
    //rand(a, b, c, d, e, f) generates a matrix with a rows and b columns in a value range of [c, d]
    salary = rand(20, 1, 250.0, 500.0, 1.0, -1);
    //with [a, b, ..] we can create a matrix with the given values.
    age = [20, 30, 23, 65, 70, 42, 34, 55, 76, 32, 53, 40, 42, 69, 63, 26, 70, 36, 21, 23];

    //createFrame() creates a Frame with the given matrices. The column names (strings) are optional.
    employee_frame = createFrame(employee_id, salary, age, "employee_id", "salary", "age");

//We register the employee_frame we created previously. note the name for the registration and the 
//name of the frame don't have to be the same.
    registerView("employee", employee_frame);

//We run a SQL Query on the registered Frame. Note here we have to reference the name we choose
//during registration.
    res = sql(
        "SELECT e.employee_id as employee_id, e.salary as salary, e.age as age
        FROM employee as e
        WHERE e.salary > 450.0;");

//We can Print both employee and the query result to the console with print().
    print(employee_frame);
    print(res);
```

### Example 2

```cpp
employee_id = seq(1, 20, 1);
salary = rand(20, 1, 250.0, 500.0, 1.0, -1);
age = [20, 30, 23, 65, 70, 42, 34, 55, 76, 32, 53, 40, 42, 69, 63, 26, 70, 36, 21, 23];

employee_frame = createFrame(employee_id, salary, age, "employee_id", "salary", "age");

registerView("employee", employee_frame);

res = sql(
    "SELECT  e.age as age, avg(e.salary) as salary
    FROM employee as e
    GROUP BY e.age
    ORDER BY e.age");

print(employee_frame);
print(res);
```
