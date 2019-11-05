'''
NOTE: A moudule only has one entity in Python.
So this params.py module would only be executed once as whole.
'''
import random
import numpy as np
from pathlib import Path
from utility import *
from numba import njit, prange

GAMMA = 0.90
BETA  = 10
STAGE = 1000
RANDOM_SEED = 58760#random.randint(0, 2**16)

N_AP  = 15
N_ES  = 10
N_JOB = 10
MQ    = 10 #maximum queue length on AP (exclusive)
LQ    = 10 #maximum queue length on ES (inclusive)

PROC_MIN  = 10
PROC_MAX  = 45
PROC_RNG  = np.arange(PROC_MIN, PROC_MAX, dtype=np.int32)
PROC_RNG_L= len(PROC_RNG)
DIM_P     = (LQ+1)*PROC_MAX

@njit
def genJobDist():
    proc_dist = np.zeros((N_ES,N_JOB,PROC_RNG_L), dtype=np.float32)
    for j in prange(N_JOB):
        for m in prange(N_ES):
            roll = np.random.randint(2)
            proc_dist[m,j] = genHeavyHeadDist(PROC_RNG_L) if roll==1 else genHeavyTailDist(PROC_RNG_L)
    return proc_dist

@njit
def reg(x):
    return min(max(0,x),LQ)

np.random.seed(RANDOM_SEED)
npzfile = 'logs/{:05d}.npz'.format(RANDOM_SEED)

if Path(npzfile).exists():
    _params = np.load(npzfile)
    arr_prob    = _params['arr_prob']
    ul_prob     = _params['ul_prob']
    proc_dist   = _params['proc_dist']
    # ul_trans    = _params['ul_trans']
else:
    arr_prob = 0.1 * np.random.rand(N_AP, N_JOB).astype(np.float32)             #[0.00, 0.01] for each
    ul_prob  = 0.3 + 0.2*np.random.rand(N_AP, N_ES, N_JOB).astype(np.float32)  #[0.30, 0.50] for each

    proc_dist = genJobDist()
    # ul_trans = TransAP()

    np.savez(npzfile, **{
        'arr_prob'  :arr_prob,
        'ul_prob'   :ul_prob,
        'proc_dist' :proc_dist,
        # 'ul_trans'  :ul_trans,
    })
    pass
