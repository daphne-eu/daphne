#!/usr/bin/env python3

from daphne.context.daphne_context import DaphneContext

dc = DaphneContext()

# Create data.
X = dc.fill(123, 1, 1)                      # matrix<1x1xsi64>

# Cast.
Y1 = X.asType(dtype="scalar")               # si64
Y2 = X.asType(dtype="matrix")               # matrix<1x1xsi64>
Y3 = X.asType(vtype="ui32")                 # matrix<1x1xui32>
Y4 = X.asType(dtype="scalar", vtype="ui32") # ui32
Y5 = X.asType(dtype="matrix", vtype="ui32") # matrix<1x1xui32>

# Use the cast result (used to fail in the past).
Y1 = Y1 + 1
Y2 = Y2 + 1
Y3 = Y3 + 1
Y4 = Y4 + 1
Y5 = Y5 + 1

# Print the results.
Y1.print().compute()
Y2.print().compute()
Y3.print().compute()
Y4.print().compute()
Y5.print().compute()