import tensorflow as tf
import numpy as np

def GetRandomBatch(x,y,count):
    assert x.shape[0] == y.shape[0],('x.shape: %s y.shape: %s' % (x.shape, y.shape))
    if count > x.shape[0]:
        count=x.shape[0]
    _x = x
    _y = y
    perm = np.arange(x.shape[0])
    np.random.shuffle(perm)
    _x = x[perm]
    _y = y[perm]
    return _x[0:count], _y[0:count]


def GetInputData():
    x=np.loadtxt('train_forces_sf.dat')
    _y=np.loadtxt('train_forces.dat')
    n=len(_y)
    y=_y.reshape((n,1))
    return x,y 



