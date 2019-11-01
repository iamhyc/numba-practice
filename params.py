'''
NOTE: A moudule only has one entity in Python.
So this params.py module would only be executed once as whole.
'''
import random
import numpy as np
from pathlib import Path

GAMMA = 0.90
BETA  = 10
STAGE = 1000
RANDOM_SEED = random.randint(2**16)

N_AP  = 15
N_ES  = 10
N_JOB = 10
MQ    = 10 #maximum queue length on AP (exclusive)
LQ    = 10 #maximum queue length on ES (inclusive)

PROC_MIN  = 10
PROC_MAX  = 45
PROC_RNG  = np.arange(PROC_MIN, PROC_MAX, dtype=np.int32)
DIM_P     = (LQ+1)*PROC_MAX

np.random.seed(RANDOM_SEED)
npzfile = 'logs/{}.npz'.format()

if Path(npzfile).exists():
    _params = np.load(npzfile)
    arr_prob    = _params['arr_prob']
    ul_prob     = _params['ul_prob']
    proc_dist   = _params['proc_dist']
    ul_trans    = _params['ul_trans']
    off_trans   = _params['off_trans']
else:
    arr_prob  = np.zeros((N_AP, N_JOB),             dtype=np.float32) #FIXME: not initialized
    ul_prob   = np.zeros((N_AP, N_ES, N_JOB),       dtype=np.float32) #FIXME: not initialized
    proc_dist = np.zeros((N_ES, N_JOB),             dtype=np.float32) #FIXME: not initialized

    ul_trans  = np.zeros((N_AP,N_ES,N_JOB, MQ,MQ),  dtype=np.float32) #FIXME: not initialized
    off_trans = np.zeros((N_ES,N_JOB,DIM_P,DIM_P),  dtype=np.float32) #FIXME: not initialized

    np.savez(npzfile, **{
        'arr_prob'  :arr_prob,
        'ul_prob'   :ul_prob,
        'proc_dist' :proc_dist,
        'ul_trans'  :ul_trans,
        'off_trans' :off_trans
    })
    pass

@njit
def reg(x):
    return min(max(0,x),LQ)