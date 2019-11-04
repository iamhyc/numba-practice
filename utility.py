
import numpy as np
from scipy.stats import norm
from numba import njit, jit

@njit
def multoss(p_vec):
    return (np.random.rand() > np.cumsum(p_vec)).argmin()

@njit
def toss(p):
    p_vec = np.array([1-p, p], dtype=np.float32)
    return multoss(p_vec)

@njit
def FillMatDiagonal(mat, arr, offset=0):
    assert(mat.shape[0] == mat.shape[1]) #assert square matrix
    for i in range(len(arr)):
        if offset > 0:
            mat[i,i+offset] = arr[i]
        else:
            mat[i+offset, i] = arr[i]
    pass

@njit
def FillAColumn(mat, idx, arr, offset=0):
    for i in range(len(arr)):
        mat[idx,i+offset] = arr[i]
    pass

@njit
def FillARow(mat, idx, arr, offset=0):
    for i in range(len(arr)):
        mat[i+offset,idx] = arr[i]
    pass

@njit
def genFlatDist(size):          #e.g. [1, 1, ... 1, 1]
    arr = 0.1+0.1*np.random.rand(size).astype(np.float32)
    return (arr / np.sum(arr))

@njit
def genHeavyTailDist(size):     #e.g. [0, 0, ... 1, 1]
    mid_size = size//2
    arr_1 = 0.1*np.random.rand(mid_size).astype(np.float32)
    arr_2 = 0.5+0.1*np.random.rand(size-mid_size).astype(np.float32)
    arr = np.sort( np.concatenate((arr_1, arr_2)) )
    return (arr / np.sum(arr))

@njit
def genHeavyHeadDist(size):     #e.g. [1, 1, ... 0, 0]
    arr = genHeavyTailDist(size)
    return arr[::-1]

def genGaussianDist(size):      #e.g. [0, 0, ..1,1,1.., 0, 0]
    rv = norm(loc=size//2, scale=0.8)
    arr = rv.pmf(np.arange(size))
    arr = np.array(arr, dtype=np.float32)
    return (arr / np.sum(arr))

def genSplitDist(size):         #e.g. [1, 1, ..0,0,0.., 1, 1]
    arr = genGaussianDist(size)
    arr = 1 - arr
    return (arr / np.sum(arr))
