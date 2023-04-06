
import numpy as np 

def Is(S) :
    return(np.argmax(np.max(S, axis=0) - np.min(S, axis=0)))
