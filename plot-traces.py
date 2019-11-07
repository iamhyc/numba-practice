
import numpy as np
import matplotlib. pyplot as plt

LOG_NUMBER = '57718'

npzfiles = 'traces-%s/{:04d}.npz'%(LOG_NUMBER)
random_cost  = np.zeros(1000, dtype=np.float32)
greedy_cost  = np.zeros(1000, dtype=np.float32)
bs_proc_cost = np.zeros(1000, dtype=np.float32)
bs_ul_cost   = np.zeros(1000, dtype=np.float32)
mdp_cost     = np.zeros(1000, dtype=np.float32)

for i in range(1000):
    trace = np.load(npzfiles.format(i))

    random_cost  = np.sum(trace['random_ap_stat'])  + np.sum(trace['random_es_stat'][:,:,0]) 
    greedy_cost  = np.sum(trace['greedy_ap_stat'])  + np.sum(trace['greedy_es_stat'][:,:,0]) 
    bs_proc_cost = np.sum(trace['bs_proc_ap_stat']) + np.sum(trace['bs_proc_es_stat'][:,:,0])    
    bs_ul_cost   = np.sum(trace['bs_ul_ap_stat'])   + np.sum(trace['bs_ul_es_stat'][:,:,0]) 
    mdp_cost     = np.sum(trace['mdp_ap_stat'])     + np.sum(trace['mdp_es_stat'][:,:,0]) 

plt.plot(range(1000), random_cost, '-ro')
plt.plot(range(1000), greedy_cost, '-ro')
plt.plot(range(1000), bs_proc_cost, '-ro')
plt.plot(range(1000), bs_ul_cost, '-ro')
plt.plot(range(1000), mdp_cost, '-ro')
plt.show()