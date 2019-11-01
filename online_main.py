
import numpy as np
from numba import njit, prange
from mdp import *
from utility import *
from params import *

@njit
def NextState(stat, policy):
    newStat = State().clone(stat)

    # toss job arrival on AP
    arrivals = np.zeros((N_AP, N_ES, N_JOB), dtype=np.int32)
    for j in prange(N_JOB):
        for k in prange(N_AP):
            arrivals[k, policy[k,j], j] = toss(arr_prob[k,j]) #m = policy[k,j]
    
    # count uploading & offloading jobs
    off_numbers = np.zeros((N_ES, N_JOB), dtype=np.int32)
    for j in prange(N_JOB):
        for m in prange(N_ES):
            for k in prange(N_AP):
                tmp_arr = [ toss(ul_prob[k,m,j]) for _ in range(stat.ap_stat[k,m,j]) ]
                newStat.ap_stat[k,m,j] = tmp_arr.count(0) + arrivals[k,m,j]
                off_numbers           += tmp_arr.count(1)
    newStat.ap_stat = np.clip(newStat.ap_stat, 0, MQ) #NOTE: CLIP [0, MQ]

    # process jobs on ES
    newStat.es_stat[:,:,0] += off_numbers       # appending new arrival jobs
    newStat.es_stat[:,:,1] -= 1                 # remaining time -1
    for j in prange(N_JOB):                     #
        for m in prange(N_ES):                  #
            if newStat.es_stat[m,j,1] <= 0:     # if first job finished:
                if newStat.es_stat[m,j,0] > 0:  #   if has_next_job:
                    newStat.es_stat[m,j,0] -= 1 #       next job join processing
                    newStat.es_stat[m,j,1] = PROC_RNG[ multoss(proc_dist[m,j]) ]
                else:                           #   else:
                    newStat.es_stat[m,j,1] = 0  #       clip the remaining time
            else:                               # else:
                pass                            #    nothing happened
    newStat.es_stat[:,:,0] = np.clip(newStat.es_stat[:,:,0], 0, LQ) #NOTE: CLIP [0,LQ]

    return newStat

@njit
def main():
    stat = State()
    stage = 0

    while stage < STAGE:
        policy = optimize(stat)
        stat = NextState(stat, policy)
        stage += 1
        pass
    pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        exit()