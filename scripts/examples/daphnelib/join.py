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

from daphne.context.daphne_context import DaphneContext
import pandas as pd

# Initialize the Daphne Context.
dc = DaphneContext()

# Customers DataFrame.
# Numerical representation of company names and contacts.
customers_df = pd.DataFrame({
    "CustomerID": [101, 102, 103],
    "CompanyName": [1, 2, 3],
    "ContactName": [1, 2, 3]   
})

# Orders DataFrame.
# Numerical representation of order dates.
orders_df = pd.DataFrame({
    "OrderID": [10643, 10692, 10702, 10704, 10705, 10710, 10713, 10715],
    "CustomerID": [101, 101, 102, 101, 102, 103, 103, 101],
    "OrderDate": [20230715, 20230722, 20230725, 20230726, 20230728, 20230730, 20230730, 20230731]
})

# Print inputs.
print("Input data frames:\n")
print("Customers:")
print(customers_df)
print("\nOrders:")
print(orders_df)

# Create DAPHNE Frames.
customers_frame = dc.from_pandas(customers_df)
orders_frame = dc.from_pandas(orders_df)

# Calculate and print the result.
print("\nResult of the join:")
join_result = customers_frame.innerJoin(orders_frame, "CustomerID", "CustomerID")
print(join_result.compute())