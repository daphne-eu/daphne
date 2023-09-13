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

print("\n\n###\n### Daphne SQL Test in Python\n###\n")
print("Initial Data Frames:\n")
print("Customers:")
print(customers_df)

# Create Daphne Frames
customers_frame = dc.from_pandas(customers_df)

customers_table = customers_frame.registerView("Customers")

# SQL Example 1: Simple SELECT query
query1 = "SELECT c.CustomerID, c.CompanyName, c.ContactName FROM Customers as c WHERE c.CustomerID = 101;"
result_frame1 = dc.sql(query1)
output_result1 = result_frame1.compute_sql([customers_table])

print(f"\nResult of running the SQL Query:\n\n{query1}\n")
print(output_result1)