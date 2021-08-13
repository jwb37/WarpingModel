import math

def calc(in_dim, k, stride, pad, dil=1 ):
    return math.floor( (in_dim + 2*pad - dil*(k-1)-1)/stride + 1)
