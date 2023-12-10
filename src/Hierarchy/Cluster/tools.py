
import numpy as np 

def Is(S) :
    if ( np.max(S[:,:,0]) - np.min(S[:,:,0]) ) < ( np.max(S[:,:,1]) - np.min(S[:,:,1]) ):
        return( 1 )
    return( 0 )
