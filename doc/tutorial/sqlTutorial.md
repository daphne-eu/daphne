# Builtin SQL
DAPHNE supports a rudimentary version of SQL. At any Point in a DAPHNE script, we can execute a SQL Query on frames.
We need two operations to achieve this: ```registerView(...)``` and ```sql(...)```

For the following examples we assume we already have a DAPHNE script which includes calculations on a frame "x" that has the columns "a", "b" and "c".

### registerView(...)
RegisterView registers a frame for the sql operation.
If we want to execute a SQL Query on a frame, we NEED to register it before that.
The operation has two inputs: the name of the table, as a string, and the frame which shall be associated with the given name.

For example, we can register the frame "x", from previous calculations, under the name "Table1". The DAPHNE script for this would look like this:
```
registerView("Table1", x);
```

### sql(...)
Now that we have registered the tables, that we need for our SQL Query, we can go ahead and execute our Query. The SQL Operation takes one input: the SQL Query, as a string. In it, we will reference the table names we previously have registered via registerView(...). As A result of this Operation, we get back a frame. The columns of the frame are named after the projection arguments inside the SQL Query.

For example, we want to return all the rows of the Frame x, which we have previously registered under the name "Table1", where the column "a" is bigger than 5 and save it in a new Frame named "y". The DAPHNE script for this would look like this:
```
y = sql("SELECT t.a as a, t.b as b, t.c as c FROM Table1 as t WHERE t.a > 5;");
```

This results in a frame "y" that has three columns "a", "b" and "c".
On the frame y we can continue to build our DAPHNE script.

### Features
We don't support the complete SQL standard at the moment. For instance, we need to fully specify on which columns we want to operate on. In the Example above, we see "t.a" instead of simply "a".
Also, not supported are DDL and DCL Queries. Our goal for DML Queries is to only support SELECT-Statements.
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
* All Set Operations (Union, Except Intersect)
* Recursive SQL Queries
* Limit
* Distinct
