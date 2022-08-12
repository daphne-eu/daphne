# Builtin SQL


### registerView()
The sql operation uses internally its own parser. This is one reason, the sql operation can't use the variable name declared before. To transfer the Frames to the sql operations, we therefor need another operation. This operation is called registerView. It takes first a String (table name) and then a variable (previously declared). This registers the Frame as a Table for the sql operations.

Simple Example:
```
// Creating a Frame with a column "i" filled with the values 1-5.
ints = frame([1,2,3,4,5], "i");

// Registrating the Frame as SQL Table "integers".
registerView("integers", ints);
```

### sql()
The Operation "sql" takes a String as an Argument. The String is for now a Simple SQL Query.
Supported Features are:
  * Cartesian Product
  * inner Join
  * Where with complex expressions
  * Order By
  * Group By
  * Having

</br></br>
***Important Notice***
  * *The Star Operator (\*) is not supported right now.*
  * *At the moment, Column names must be fully qualified. (tablename.column_name instead of just column_name)*
  * *Daphne doesn't support implicit casting. So the base Datatype must be the same for arithmetic operations, comparisons...*

Simple Example:
```
// Filter the Frame using SQL for integers bigger than two
result = sql(SELECT t.i1 FROM integers t WHERE t.i > 2);
```

### HEADING
In DAPHNE we can read in Frames from csv files.
With the sql operation, we can now filter join and group these frames.
The result could be written back into a csv file or be used for machine learning (ML).

Example:
```


```
