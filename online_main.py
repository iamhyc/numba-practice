
import pathlib
import numpy as np
from numba import njit, prange
from mdp import *
from utility import *
from params import *
import matplotlib.pyplot as plt

def NextState(stat, policy, arrival_ap):
    newStat = State().clone(stat)

    # toss job arrival on AP
    arrivals = np.zeros((N_AP, N_ES, N_JOB), dtype=np.int32)
    for j in range(N_JOB):
        for k in range(N_AP):
            arrivals[k, policy[k,j], j] = arrival_ap[k,j] #m = policy[k,j]

    # count uploading & offloading jobs
    off_numbers = np.zeros((N_ES, N_JOB), dtype=np.int32)
    for j in range(N_JOB):
        for m in range(N_ES):
            for k in range(N_AP):
                newStat.ap_stat[k,m,j] = 0 #to copy from stat
                for _ in range(stat.ap_stat[k,m,j]):
                    tmp = toss(ul_prob[k,m,j])
                    newStat.ap_stat[k,m,j] += tmp
                    off_numbers[m,j]       += tmp
                    pass
                newStat.ap_stat[k,m,j] += arrivals[k,m,j]
                if newStat.ap_stat[k,m,j] >= MQ:
                    newStat.ap_stat[k,m,j] = MQ-1

                # tmp_arr = [ toss(ul_prob[k,m,j]) for _ in range(stat.ap_stat[k,m,j]) ] #NOTE: AVOID list compression
                # # print(tmp_arr)
                # newStat.ap_stat[k,m,j] = tmp_arr.count(0) + arrivals[k,m,j]
                # if newStat.ap_stat[k,m,j] >= MQ:    #NOTE: CLIP [0, MQ]
                #     newStat.ap_stat[k,m,j] = MQ-1
                # off_numbers[m,j]      += tmp_arr.count(1)

    # print(off_numbers)
    # process jobs on ES
    #NOTE: AVOID such operation when using prange!!!
    # newStat.es_stat[:,:,0] += off_numbers       # appending new arrival jobs
    # newStat.es_stat[:,:,1] -= 1                 # remaining time -1

    for j in range(N_JOB):
        for m in range(N_ES):
            newStat.es_stat[m,j,0] += off_numbers[m,j]
            newStat.es_stat[m,j,1] -= 1

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

def test():
    pathlib.Path('./traces-{:05d}'.format(RANDOM_SEED)).mkdir(exist_ok=True)

    stage = 0
    bs_policy = BaselinePolicy()
    rd_stat, bs_stat, mdp_stat = State(), State(), State()
    (y1, y2, y3) = (rd_stat, bs_stat, mdp_stat)

    while stage < STAGE:
        # toss job arrival on AP
        arrival_ap = np.zeros((N_AP, N_JOB), dtype=np.int32)
        for j in range(N_JOB):
            for k in range(N_AP):
                arrival_ap[k,j] = toss(arr_prob[k,j]) #m = policy[k,j]

        rd_policy     = RandomPolicy()
        mdp_policy, val = optimize(mdp_stat)

        mdp_stat = NextState(mdp_stat, mdp_policy, arrival_ap)
        bs_stat  = NextState(bs_stat,  bs_policy,  arrival_ap)
        rd_stat  = NextState(rd_stat,  rd_policy,  arrival_ap)

        plt.plot([stage, stage+1], [y1.cost(), rd_stat.cost()],  'co-')
        plt.plot([stage, stage+1], [y2.cost(), bs_stat.cost()],  'ko-')
        plt.plot([stage, stage+1], [y3.cost(), mdp_stat.cost()], 'ro-')
        plt.pause(0.05)
        (y1, y2, y3) = (rd_stat, bs_stat, mdp_stat)

        trace_file = 'traces-{:05d}/{:04d}.npz'.format(RANDOM_SEED, stage)
        np.savez(trace_file, **{
            'mdp_ap_stat': mdp_stat.ap_stat,
            'mdp_es_stat': mdp_stat.es_stat,
            'bs_ap_stat':  bs_stat.ap_stat,
            'bs_es_stat':  bs_stat.es_stat,
            'rd_ap_stat':  rd_stat.ap_stat,
            'rd_es_stat':  rd_stat.es_stat,
            'mdp_value':   val
        })

        stage += 1
        pass

    plt.show()
    pass

def main():
    pathlib.Path('./logs').mkdir(exist_ok=True)
    pathlib.Path('./figures').mkdir(exist_ok=True)
    pathlib.Path('./traces').mkdir(exist_ok=True)

    stat = State()
    stage = 0
    (x1, y1) = (0, 0)
    plt.autoscale(True)

    while stage < STAGE:
        policy, val = optimize(stat)
        stat = NextState(stat, policy)
        
        trace_file = 'traces/{:4d}.npz'.format(stage)
        np.savez(trace_file, **{
            'ap_stat': stat.ap_stat,
            'es_stat': stat.es_stat,
            'policy' : policy,
            'value'  : val
        })
        #print('Stage: {stage} \n Policy: {policy} \n Value: {value} \n'.format(
        #    stage=stage, policy=policy, value=val
        #))

        (x2, y2) = (stage, np.sum(val))
        plt.subplot(1, 2, 1)
        plt.plot([x1, x2], [y1, y2], 'ko-')
        plt.subplot(1, 2, 2)
        plt.scatter(stage, np.sum(stat.es_stat[:,:,0]), c='black')
        # print(stat.es_stat[:,:,1])
        (x1, y1) = (x2, y2)
        plt.pause(0.05)

        stage += 1
        pass

    plt.show()
    pass

test()
# main()
# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         raise e
#     finally:
#         exit()