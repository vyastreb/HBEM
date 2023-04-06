
import numpy as np 
import numpy.linalg as lin 


def x_gap(X, Y) :
    """
    x_gap(X, Y) 
        -- Gap between two points X, Y
    """
    return(lin.norm(X - Y))


def Size_def(S_ex):
	return( lin.norm( S_ex[:,0] - S_ex[:,1] ) )

