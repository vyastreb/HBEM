

import numpy as np 
import numpy.linalg as lin 


# def ArgMaxM(M, P) :

#     pivot, i_s = 0, 0
#     for i in range(len(M)) :
#         if i not in P :
#             if pivot < np.abs(M[i][0]) :
#                 pivot = np.abs(M[i][0])
#                 i_s = i
#     return( M[i_s][0], i_s )


def ArgMaxV(V, P) : 

    pivot, i_s = 0, 0
    for i in range(len(V)) :
        if i not in P :
            if pivot < np.abs(V[i]) :
                pivot = np.abs(V[i])
                i_s = i
    return( V[i_s], i_s )

# def ArgMinM(M, P) :

#     pivot, i_s = M[0][0], 0
#     for i in range(1,len(M)) :
#         if i not in P :
#             if pivot > np.abs(M[i][0]) :
#                 pivot = np.abs(M[i][0])
#                 i_s = i
#     return( M[i_s][0], i_s )


# def ArgMinV(V, P) : 

#     pivot, i_s = V[0], 0
#     for i in range(1,len(V)) :
#         if i not in P :
#             if pivot > np.abs(V[i]) :
#                 pivot = np.abs(V[i])
#                 i_s = i
#     return( V[i_s], i_s )


def ArgMin(n, P) :
    
    i = 0
    while i < n :
        if i not in P :
            return( i )
        i += 1
    return( None )


def Residual(A_, u_) :
    return( np.dot(u_.reshape(1,u_.shape[0]), A_) )