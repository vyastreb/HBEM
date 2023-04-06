
import numpy as np 
import numpy.linalg as lin

def x_gap(X, Y) :
    return(lin.norm(X - Y))

def elem_size_def(elem) :
    return(np.max(lin.norm(elem[0,:] - elem[1:,:], axis=1)))

