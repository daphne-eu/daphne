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

print("\n\n###\n### Daphne SQL and Join Test in Python\n###\n")
print("Initial Data Frames:\n")
print("Customers:")
print(customers_df)
print("\nOrders:")
print(orders_df)

# Create Daphne Frames
customers_frame = dc.from_pandas(customers_df)
orders_frame = dc.from_pandas(orders_df)

customers_table = customers_frame.registerView("Customers")
orders_table = orders_frame.registerView("Orders")

# SQL Example 2: Inner Join with SQL
query2 = "SELECT c.CompanyName, o.OrderID FROM Customers as c INNER JOIN Orders as o ON c.CustomerID = o.CustomerID;"
result_frame2 = dc.sql(query2)
output_result2 = result_frame2.compute_sql([customers_table, orders_table])

print(f"\nResult of running the SQL Query:\n\n{query2}\n")
print(output_result2)