import numpy as np

# naive implementation
np.random.seed(0)
rs = []
for n in range(1, 11):
    r = np.random.rand()
    rs.append(r)
    q = sum(rs) / n
    print(q)

print('---')

# incremental implementation
np.random.seed(0)
q = 0
for n in range(1, 11):
    r = np.random.rand()
    q = q + (r - q) / n
    print(q)
