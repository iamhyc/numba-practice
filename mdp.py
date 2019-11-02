
import numpy as np
from numpy import arange
from numba import njit, prange, jitclass
from numba import int32, float32
from numba.typed import Dict
from itertools import product
from params import *

@jitclass([
    ('ap_stat', int32[:,:,:]),
    ('es_stat', int32[:,:,:])
])
class State(object):
    def __init__(self):
        self.ap_stat = np.zeros((N_AP, N_ES, N_JOB), dtype=np.int32)
        self.es_stat = np.zeros((N_ES, N_JOB, 2),    dtype=np.int32)
        pass

    def clone(self, stat):
        self.ap_stat = np.copy(stat.ap_stat)
        self.es_stat = np.copy(stat.es_stat)
        return self
    pass

@njit
def Policy():
    return np.zeros((N_AP,N_JOB), dtype=np.int32)

@njit
def RandomPolicy():
    return np.zeros((N_AP,N_JOB), dtype=np.int32) #FIXME: not this

@njit
def BaselinePolicy():
    return np.zeros((N_AP,N_JOB), dtype=np.int32) #FIXME: not this

@jit(forceobj=True)
def cartesian(*arrays):
    tmp = list(product(*arrays))
    return np.array(tmp)

MatEntryL = cartesian(arange(LQ+1), arange(PROC_MAX))
APValVec  = np.arange(MQ, dtype=np.float32)
ESValVec  = np.repeat(arange(LQ+1), repeats=PROC_MAX).astype(np.float32)

@njit
def AP2Vec(ap_stat):
    ap_vec = np.zeros(MQ, dtype=np.float32)
    ap_vec[ ap_stat ] = 1
    return ap_vec

@njit
def ES2Vec(es_stat):
    es_vec = np.zeros(DIM_P, dtype=np.float32)
    _idx   = es_stat[0]*PROC_MAX + es_stat[1]
    es_vec[_idx] = 1
    return es_vec

@njit
def ES2Entry(l,r):
    return l*PROC_MAX + r

@njit
def TransES(beta, proc_dist):
    mat = np.zeros((DIM_P, DIM_P), dtype=np.float32)
    
    # fill in r==0
    for l in range(LQ+1):
        e = ES2Entry(l,0)
        for idx,prob in enumerate(proc_dist):
            mat[e, ES2Entry(l,        PROC_RNG[idx])] = prob*beta
            mat[e, ES2Entry(reg(l-1), PROC_RNG[idx])] = prob*(1-beta)
    
    # fill in r!=0
    for l in range(LQ+1):
        for r in range(1,PROC_MAX):
            e = ES2Entry(l, r)
            mat[e, ES2Entry(reg(l+1), r-1)] = beta
            mat[e, ES2Entry(l,        r-1)] = 1-beta

    return mat

@njit
def evaluate(x0, j, stat):

    pass

@njit
def optimize(stat):
    policy = BaselinePolicy()
    for j in prange(N_JOB):
        x0    = policy[:,j]
        order = np.random.permutation(N_AP)
        for k in range(N_AP):
            val_collection = np.zeros(N_ES, dtype=np.float32)
            for m in prange(N_ES):
                x1 = np.copy(x0)
                x1[ order[k] ] = m
                val_collection[m] = evaluate(j, stat, x1)
            x0[ order[k] ] = val_collection.argmin()
        pass
    pass