from daphne.context.daphne_context import DaphneContext

dc = DaphneContext()

M = (dc.seq(1, 8) % 3) == 1

X = dc.seq(1, 8)
Y = dc.seq(1, 8) * 10

# mat ? sca : sca
M.ifElse(3.141, 2.718).print().compute()
# mat ? mat : sca
M.ifElse(X, 0).print().compute()
# mat ? sca : mat
M.ifElse(0, Y).print().compute()
# mat ? mat : mat
M.ifElse(X, Y).print().compute()