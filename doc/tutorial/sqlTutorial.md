# Using SQL in DaphneDSL
DAPHNE supports a rudimentary version of SQL. At any point in a DaphneDSL script, we can execute a SQL query on frames.
We need two operations to achieve this: ```registerView(...)``` and ```sql(...)```

For the following examples we assume we already have a DaphneDSL script which includes calculations on a frame "x" that has the columns "a", "b" and "c".

### registerView(...)
RegisterView registers a frame for the sql operation.
If we want to execute a SQL query on a frame, we *need* to register it before that.
The operation has two inputs: the name of the table, as a string, and the frame which shall be associated with the given name.

For example, we can register the frame "x", from previous calculations, under the name "Table1". The DaphneDSL script for this would look like this:
```
registerView("Table1", x);
```

### sql(...)
Now that we have registered the tables, that we need for our SQL query, we can go ahead and execute our query. The SQL operation takes one input: the SQL query, as a string. In it, we will reference the table names we previously have registered via registerView(...). As a result of this operation, we get back a frame. The columns of the frame are named after the projection arguments inside the SQL query.

For example, we want to return all the rows of the frame x, which we have previously registered under the name "Table1", where the column "a" is greater than 5 and save it in a new frame named "y". The DaphneDSL script for this would look like this:
```
y = sql("SELECT t.a as a, t.b as b, t.c as c FROM Table1 as t WHERE t.a > 5;");
```

This results in a frame "y" that has three columns "a", "b" and "c".
On the frame y we can continue to build our DaphneDSL script.

### Features
We don't support the complete SQL standard at the moment. For instance, we need to fully specify on which columns we want to operate. In the example above, we see "t.a" instead of simply "a".
Also, not supported are DDL and DCL Queries. Our goal for DML queries is to only support SELECT-statements.
Other features we do and don't support right now can be found below.

#### Supported Features
* Cross Product
* Complex Where Clauses
* Inner Join with single and multiple join conditions separated by an "AND" Operator
* Group By Clauses
* Having Clauses
* Order By Clauses
* As

#### Not Yet Supported Features
* The Star Operator \*
* Nested SQL Queries like: ```SELECT a FROM x WHERE a IN SELECT a FROM y```
* All Set Operations (Union, Except, Intersect)
* Recursive SQL Queries
* Limit
* Distinct
