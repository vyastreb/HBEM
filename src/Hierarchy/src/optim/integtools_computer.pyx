cimport numpy as cnp 

import numpy as np 
import numpy.linalg as lin

from .normmodule import norm_computer_array_optim
from .normmodule import n_computer_optim, x_computer_optim

### custom shape functions 
def custom_n_shape( n ):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty((3,))
    n_computer_optim( n, out )
    return( out )

def custom_n_array_shape( n ) :
    cdef int nbpt = len( n )
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.empty((n, 3))
    for i in range( nbpt ) :
        n_computer_optim( n[i], out[i] )
    return( out )

def custom_x_shape( n, mesh_e ) :
    cdef cnp.ndarray[cnp.float64_t] out = np.empty((2,))
    x_computer_optim( n, mesh_e.ravel(), out )
    return( out )

def custom_x_array_shape( n, mesh_e ) :
    cdef int nbpt = n.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.empty((nbpt, 2))
    cdef cnp.ndarray[cnp.float64_t] mesh_er = mesh_e.reshape(6)
    for i in range( nbpt ) :
        x_computer_optim( n[i], mesh_er, out[i] )
    return( out )

## custom norm for the kernel function
def custom_norm( X ):
    nrow, ncol = X.shape 
    cdef cnp.ndarray[cnp.float64_t] out = np.empty((nrow,))
    norm_computer_array_optim(X.ravel(), out)
    return out

