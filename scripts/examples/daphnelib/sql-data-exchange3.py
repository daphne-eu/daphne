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

# Orders DataFrame
orders_df = pd.DataFrame({
    'OrderID': [10643, 10692, 10702],
    'CustomerID': [101, 102, 103],
    'OrderDate': [20230715, 20230722, 20230725] # Numerical representation of order dates, as Strings & Dates are not supported in Daphne yet.
})

print("\n\n###\n### Daphne SQL and Join Test in Python\n###\n")
print("Initial Data Frames:\n")
print("\nOrders:")
print(orders_df)

# Create Daphne Frames
orders_frame = dc.from_pandas(orders_df)

orders_table = orders_frame.registerView("Orders")

# SQL Example 3: Group By with Having
query3 = "SELECT o.CustomerID FROM Orders as o WHERE o.OrderDate > 20230721 GROUP BY o.CustomerID;"
result_frame3 = dc.sql(query3)
output_result3 = result_frame3.compute_sql([orders_table])

print(f"\nResult of running the SQL Query:\n\n{query3}\n")
print(output_result3)