
import numpy as np
from numpy import arange
from numba import jit, njit, prange, jitclass
from numba import int32, float32
from numba.typed import Dict
from itertools import product
from params import *
from utility import *

APValVec  = np.arange(MQ, dtype=np.float32)
ESValVec  = np.repeat(arange(LQ+1), repeats=PROC_MAX).astype(np.float32)

@jitclass([ ('ap_stat', int32[:,:,:]), ('es_stat', int32[:,:,:]) ])
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
    policy = np.zeros((N_AP, N_JOB), dtype=np.int32)
    for k in prange(N_AP):
        for j in prange(N_JOB):
            policy[k,j] = np.random.randint(N_ES)
    return policy

@njit
def BaselinePolicy():
    policy = np.zeros((N_AP, N_JOB),dtype=np.int32)
    proc_rng = np.copy(PROC_RNG).astype(np.float32)
    for k in prange(N_AP):
        for j in prange(N_JOB):
            policy[k,j] = (proc_dist[:,j,:] @ proc_rng).argmin()
    return policy

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
def TransAP(arr_prob, ul_prob):
    ul_trans = np.zeros((MQ,MQ), dtype=np.float32)
    for n1 in range(MQ):
        ul_trans[n1, 0]                 = (1-arr_prob) *binom(n=n1, p=ul_prob, k=n1) #no arrival
        ul_trans[n1, min(n1+1, MQ-1)]  += arr_prob     *binom(n=n1, p=ul_prob, k=0) #no departure
        for n2 in range(1, n1+1):
            ul_trans[n1, n2] += arr_prob    * binom(n=n1, p=ul_prob, k=n1-n2+1)
            ul_trans[n1, n2] += (1-arr_prob)* binom(n=n1, p=ul_prob, k=n1-n2)
    return ul_trans

@njit
def TransES(alpha, proc_dist):
    mat = np.zeros((DIM_P, DIM_P), dtype=np.float32)
    
    # fill in l==0 && r==0
    mat[ES2Entry(0, 0), ES2Entry(0, 0)] = 1 - alpha
    for idx,prob in enumerate(proc_dist):
        mat[ES2Entry(0, 0), ES2Entry(0, PROC_RNG[idx])] = alpha*prob

    # fill in l!=0 && r==0
    for l in range(1, LQ+1):
        e = ES2Entry(l,0)
        for idx,prob in enumerate(proc_dist):
            mat[e, ES2Entry(l,   PROC_RNG[idx])] = prob*alpha
            mat[e, ES2Entry(l-1, PROC_RNG[idx])] = prob*(1-alpha)

    # fill in r!=0
    for l in prange(LQ+1):
        for r in prange(1,PROC_MAX):
            e = ES2Entry(l, r)
            l2 = LQ if l+1>LQ else (l+1)
            mat[e, ES2Entry(l2, r-1)] += alpha
            mat[e, ES2Entry(l,  r-1)] += 1-alpha

    # for l in prange(LQ+1):
    #     for r in prange(PROC_MAX):
    #         e = ES2Entry(l,r)
    #         assert( abs((mat[e, :]).sum() - 1.0) < 0.0001 )
    return mat

@njit
def evaluate(x0, j, stat):
    val_ap = np.zeros((N_AP, N_ES),     dtype=np.float32)
    val_es = np.zeros((N_ES,),          dtype=np.float32)
    ap_vec = np.zeros((N_AP,N_ES, MQ),  dtype=np.float32)
    es_vec = np.zeros((N_ES,DIM_P),     dtype=np.float32)

    # init vector
    for m in prange(N_ES):
        es_vec[m] = ES2Vec(stat.es_stat[m,j])
        for k in prange(N_AP):
            ap_vec[k,m] = AP2Vec(stat.ap_stat[k,m,j])

    # iteration and collect cost
    for n in range(100): #NOTE: could not parallel
        _alpha = np.zeros((N_ES, ),         dtype=np.float32)
        # calculate _alpha and val_ap
        for m in prange(N_ES):
            for k in prange(N_AP):
                _m = x0[k]
                mat = TransAP((_m==m)*arr_prob[k,j], ul_prob[k,m,j])
                ap_vec[k,m] = mat.T @ ap_vec[k,m]
                _alpha[m]   += (ap_vec[k,m] @ APValVec) * ul_prob[k,m,j] #NOTE: REALLY?
                val_ap[k,m] += (ap_vec[k,m] @ APValVec) * np.power(GAMMA, n)
            pass
        # calculate val_es with _alpha
        for m in prange(N_ES):
            mat = TransES(_alpha[m], proc_dist[m,j])
            es_vec[m]  = mat.T @ es_vec[m]
            val_es[m] += (es_vec[m] @ ESValVec) * np.power(GAMMA, n)
        # print(es_vec)
        pass
    
    return np.sum(val_ap) + np.sum(val_es)

@njit
def optimize(stat):
    policy = BaselinePolicy()

    for j in prange(N_JOB):
        x0    = policy[:,j]
        order = np.random.permutation(N_AP)
        for k in prange(N_AP):
            val_collection = np.zeros(N_ES, dtype=np.float32)
            for m in prange(N_ES):
                x1 = np.copy(x0)
                x1[ order[k] ] = m
                val_collection[m] = evaluate(x1, j, stat)
                pass
            policy[ order[k],j ] = val_collection.argmin()
        pass

    val_collection = np.zeros(N_JOB, dtype=np.float32)
    for j in prange(N_JOB):
        x0 = policy[:, j]
        val_collection[j] = evaluate(x0, j, stat)
    return policy, val_collection