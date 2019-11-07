
import numpy as np
import matplotlib. pyplot as plt

npzfiles = 'traces-09397/{:04d}.npz'
mdp_cost = np.zeros(1000, dtype=np.float32)
bs_cost  = np.zeros(1000, dtype=np.float32)
rd_cost  = np.zeros(1000, dtype=np.float32)

for i in range(1000):
    trace = np.load(npzfiles.format(i))
    mdp_cost[i] = np.sum(trace['mdp_ap_stat']) + np.sum(trace['mdp_es_stat'][:,:,0])
    bs_cost[i] = np.sum(trace['bs_ap_stat']) + np.sum(trace['bs_es_stat'][:,:,0])
    rd_cost[i] = np.sum(trace['rd_ap_stat']) + np.sum(trace['rd_es_stat'][:,:,0])

plt.plot(range(1000), mdp_cost, '-ro')
plt.plot(range(1000), bs_cost, '-ko')
plt.plot(range(1000), rd_cost, '-co')
plt.show()