    
import os 
import pathlib as pl 
import numpy as np

cimport numpy as cnp

path = pl.PurePath( __file__ ).parent 


def gauss_tri( ):
    """
        Function which read all the coordinates and the weights in Gauss_coord_tri_b.npz file
    """

    os.chdir( path )
    cdef cnp.ndarray[cnp.float64_t, ndim=3] C_ = np.load('Gauss_coord_tri_b.npz', allow_pickle=True)['C']
    cdef cnp.ndarray[cnp.float64_t, ndim=2] W_ = np.load('Gauss_coord_tri_b.npz', allow_pickle=True)['W']

    return( C_, W_ )

def gauss_6pt( ):
    """
        Function which selects the first row coordinates and weights in Gauss_coord_tri_b.npz
            -- return C_6pt, W_6pt
            -- C_6pt the coordinates of the 6 integration points in the triangle of reference
            -- W_6pt the corresponding wiegths 
    """
 
    cdef cnp.ndarray[cnp.float64_t, ndim=3] C_ 
    cdef cnp.ndarray[cnp.float64_t, ndim=2] W_ 
    C_, W_ = gauss_tri( )

    cdef cnp.ndarray[ cnp.float64_t, ndim=2] C_6pt = C_[0][:6]
    cdef cnp.ndarray[ cnp.float64_t, ndim=1] W_6pt = W_[0][:6]

    del C_, W_

    return( C_6pt, W_6pt )

def gauss_quad( ) :
    """
        Function which read the Gauss_coord_quad_b.npz 
            -- same procedure as for gauss_tri
    """
    os.chdir( path )
    cdef cnp.ndarray[cnp.float64_t, ndim=2] C_ = np.load('Gauss_coord_quad_b.npz', allow_pickle=True)['C']
    cdef cnp.ndarray[cnp.float64_t, ndim=2] W_ = np.load('Gauss_coord_quad_b.npz', allow_pickle=True)['W']

    return( C_, W_ )

