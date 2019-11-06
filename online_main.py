
import numpy as np
from numba import njit, prange
from mdp import *
from utility import *
from params import *
import matplotlib.pyplot as plt
from itertools import product

def NextState(stat, policy):
    newStat = State().clone(stat)

    # toss job arrival on AP
    arrivals = np.zeros((N_AP, N_ES, N_JOB), dtype=np.int32)
    for j in range(N_JOB):
        for k in range(N_AP):
            arrivals[k, policy[k,j], j] = toss(arr_prob[k,j]) #m = policy[k,j]
    
    # count uploading & offloading jobs
    off_numbers = np.zeros((N_ES, N_JOB), dtype=np.int32)
    for j in range(N_JOB):
        for m in range(N_ES):
            for k in range(N_AP):
                tmp_arr = [ toss(ul_prob[k,m,j]) for _ in range(stat.ap_stat[k,m,j]) ]
                newStat.ap_stat[k,m,j] = tmp_arr.count(0) + arrivals[k,m,j]
                if newStat.ap_stat[k,m,j] >= MQ:    #NOTE: CLIP [0, MQ]
                    newStat.ap_stat[k,m,j] = MQ-1
                off_numbers[m,j]      += tmp_arr.count(1)

    print(off_numbers)
    # process jobs on ES
    newStat.es_stat[:,:,0] += off_numbers       # appending new arrival jobs
    newStat.es_stat[:,:,1] -= 1                 # remaining time -1
    for j in range(N_JOB):                     #
        for m in range(N_ES):                  #
            if newStat.es_stat[m,j,0] > LQ:     # NOTE: CLIP [0,LQ]
                newStat.es_stat[m,j,0] = LQ     #
            if newStat.es_stat[m,j,1] <= 0:     # if first job finished:
                if newStat.es_stat[m,j,0] > 0:  #   if has_next_job:
                    newStat.es_stat[m,j,0] -= 1 #       next job join processing
                    newStat.es_stat[m,j,1] = PROC_RNG[ multoss(proc_dist[m,j]) ]
                else:                           #   else:
                    newStat.es_stat[m,j,1] = 0  #       clip the remaining time
            else:                               # else:
                pass                            #    nothing happened

    return newStat


def main():
    stat = State()
    # for m,j in product(range(N_ES), range(N_JOB)):
        # stat.es_stat[m,j,0] = 2

    stage = 0
    plt.autoscale(True)

    (x1, y1) = (0, 0)
    tmp = stat.es_stat

    while stage < STAGE:
        policy, val = optimize(stat)
        stat = NextState(stat, policy)
        
        #print('Stage: {stage} \n Policy: {policy} \n Value: {value} \n'.format(
        #    stage=stage, policy=policy, value=val
        #))
        (x2, y2) = (stage, np.sum(val))

        tmp = stat.es_stat

        plt.subplot(1, 2, 1)
        plt.plot([x1, x2], [y1, y2], 'ko-')
        plt.subplot(1, 2, 2)
        plt.scatter(stage, np.sum(stat.es_stat[:,:,0]), c='black')
        print(stat.es_stat[:,:,1])
        (x1, y1) = (stage, np.sum(val))
        plt.pause(0.05)

        stage += 1
        pass
    plt.show()
    pass

main()
# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         raise e
#     finally:
#         exit()