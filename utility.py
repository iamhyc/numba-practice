
import numpy as np
from numba import njit

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
def genUniformDist():       #e.g. [1, 1, ... 1, 1]
    pass

@njit
def genHeavyTailDist():     #e.g. [0, 0, ... 1, 1]
    pass

@njit
def genHeavyHeadDist():     #e.g. [1, 1, ... 0, 0]
    pass

@njit
def genSplitDist():         #e.g. [1, 1, ..0,0,0.. 1, 1]
    pass