# Copyright 2023 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from api.python.context.daphne_context import DaphneContext
import pandas as pd

# Initialize the Daphne Context
dc = DaphneContext()

# Customers DataFrame
customers_df = pd.DataFrame({
    'CustomerID': [101, 102, 103],
    'CompanyName': [1, 2, 3], # Numerical representation of company names, as Strings are not supported in Daphne yet.
    'ContactName': [1, 2, 3]   
})

# Orders DataFrame
orders_df = pd.DataFrame({
    'OrderID': [10643, 10692, 10702],
    'CustomerID': [101, 102, 103],
    'OrderDate': [20230715, 20230722, 20230725] # Numerical representation of order dates, as Strings & Dates are not supported in Daphne yet.
})

print("\n\n###\n### Daphne Join Test in Python\n###\n")
print("Initial Data Frames:\n")
print("Customers:")
print(customers_df)
print("\nOrders:")
print(orders_df)

# Create Daphne Frames
customers_frame = dc.from_pandas(customers_df)
orders_frame = dc.from_pandas(orders_df)

print("\nResult of the first Join:")
# Join Example 1: Inner Join with Daphne method
join_result1 = customers_frame.innerJoin(orders_frame, "CustomerID", "CustomerID")
print(join_result1.compute())


""" Both Semi Join and Group Join are not working in Daphne yet. 
This can be reproduced in the file "issue-with-joins.daph"
They Return two values: 

Returned Value 1: 
Frame(3x3, [CustomerID:int64_t, CompanyName:int64_t, ContactName:int64_t])
101 1 1
102 2 2
103 3 3
Frame(3x3, [OrderID:int64_t, CustomerID:int64_t, OrderDate:int64_t])
10643 101 20230715
10692 102 20230722
10702 103 20230725
Frame(3x2, [l.CustomerID:int64_t, SUM(r.OrderID):int64_t])
103 10702
102 10692
101 10643

Returned Value 2: 
DenseMatrix(3x1, uint64_t)
2
1
0

print("\nResult of the second Join:")
# Join Example 2: Semi Join
print(customers_frame.semiJoin(orders_frame, "CustomerID", "CustomerID").compute())
#print(join_result2.compute())

# Join Example 3: Group Join
join_result3 = customers_frame.groupJoin(orders_frame, "CustomerID", "CustomerID", "OrderID")
print(join_result3.compute())
"""