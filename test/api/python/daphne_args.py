import numpy as np
from daphne.context.daphne_context import DaphneContext

m1 = np.array([3, 9, 12])
dc = DaphneContext()
X = dc.from_numpy(m1)
X.print().compute(daphne_args=["--vec", "--pin-workers"])
X.print().compute(daphne_args=["--explain", "parsing_simplified"])
X.print().compute(daphne_args=["--explain", "parsing_simplified,parsing", "--timing"])
X.print().compute(daphne_args="--explain=parsing_simplified,parsing --timing")