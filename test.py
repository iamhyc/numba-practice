
from numba import njit, prange
from functools import reduce
import numpy as np

@njit
def test(n):
    a_b = reduce(lambda x,y:x*y, range(1,n+1))
    return a_b

print( test(10) )