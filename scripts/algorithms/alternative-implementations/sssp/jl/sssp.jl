using MatrixMarket


G = mmread(ARGS[1])
n = size(G, 1)
x = zeros(n)
x[1] = 1.0
distances = fill(Inf, n)
distances[1] = 0.0

y = G * x # voisins de start
y = min.(ones(n), y) # we replace by ones the activated neighbors

z = 2 * min.(ones(n), G * y) # the 2nd rank neighbors
y = y + z




