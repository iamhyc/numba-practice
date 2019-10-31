
import numpy as np
from numba import njit

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