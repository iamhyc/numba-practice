import random
import numpy as np
from scipy.interpolate import interp1d
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
    random_cost[i]  = np.sum(trace['random_ap_stat'])  + np.sum(trace['random_es_stat'][:,:,0]) 
    greedy_cost[i]  = np.sum(trace['greedy_ap_stat'])  + np.sum(trace['greedy_es_stat'][:,:,0]) 
    bs_proc_cost[i] = np.sum(trace['bs_proc_ap_stat']) + np.sum(trace['bs_proc_es_stat'][:,:,0])    
    bs_ul_cost[i]   = np.sum(trace['bs_ul_ap_stat'])   + np.sum(trace['bs_ul_es_stat'][:,:,0]) 
    mdp_cost[i]     = np.sum(trace['mdp_ap_stat'])     + np.sum(trace['mdp_es_stat'][:,:,0])
    pass

def plot_cost_vs_time():
    plt.plot(range(1000), random_cost,  '-co')
    plt.plot(range(1000), greedy_cost,  '-ko')
    plt.plot(range(1000), bs_proc_cost, '-bo')
    plt.plot(range(1000), bs_ul_cost,   '-go')
    plt.plot(range(1000), mdp_cost,     '-ro')
    plt.legend(['Random Policy', 'Greedy Policy', 'Baseline (processing time)', 'Baseline (uploading time)', 'MDP Policy'])
    plt.show()
    pass

def plot_cost_cdf_cmp_algorithm():
    y = [0] * 5
    cost_scale = 1
    y[0] = np.sort(random_cost / cost_scale)
    y[1] = np.sort(greedy_cost / cost_scale)
    y[2] = np.sort(bs_proc_cost/ cost_scale)
    y[3] = np.sort(bs_ul_cost  / cost_scale)
    y[4] = np.sort(mdp_cost    / cost_scale)
    ylim = max([item.max() for item in y])

    x     = np.linspace(0, ylim, num=1000)
    y_pmf = np.zeros((5, 1000))
    inter_x     = np.linspace(0, ylim, num=200)
    inter_y_pmf = np.zeros((5, 200))

    for i in range(5):
        for j in range(1,1000):
            y_pmf[i][j] = np.logical_and(y[i]>=x[j-1], y[i]<x[j]).sum()
        y_pmf[i] = np.cumsum(y_pmf[i]) / 1000
        interp_func = interp1d(x, y_pmf[i], kind='quadratic')
        inter_y_pmf[i] = interp_func(inter_x)

    plt.xlim(0, ylim)
    # plt.ylim(0, 1.0)
    plt.plot(inter_x, inter_y_pmf[0], '-c')
    plt.plot(inter_x, inter_y_pmf[1], '-k')
    plt.plot(inter_x, inter_y_pmf[2], '-b')
    plt.plot(inter_x, inter_y_pmf[3], '-g')
    plt.plot(inter_x, inter_y_pmf[4], '-r')
    plt.legend(['Random Policy', 'Greedy Policy', 'Baseline (processing time)', 'Baseline (uploading time)', 'MDP Policy'])
    plt.show()
    pass

# plot_cost_vs_time()
plot_cost_cdf_cmp_algorithm()