
import numpy as np
from numba import njit, prange, jitclass
from numba import int32, float32
from numba.typed import Dict
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

@njit
def AP2Vec(ap_stat):
    pass

@njit
def ES2Vec(es_stat):
    pass

@njit
def ES2Entry(l,r):
    pass

@njit
def evaluation():
    pass

@njit
def optimize():
    pass