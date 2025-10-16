import numpy as np

dm_adj = np.zeros((64, 64), dtype=int)


with open('data/dm_unit_adj_m.txt') as f:
    contents = f.read()

for line in contents.split('\n'):
    try:
        a, adj_list = line.split('\t')
        for b in adj_list.split(' '):
            print(f"{a}->{b}")
            _a,_b = int(a)-1,int(b)-1
            dm_adj[_a,_b] = 1
            dm_adj[_b,_a] = 1
    except:
        continue

np.savetxt('dm_adj.txt', dm_adj, fmt="%d")