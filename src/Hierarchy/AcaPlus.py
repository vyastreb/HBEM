

import numpy as np 
import numpy.linalg as lin
from .AcaTools import Residual, ArgMin, ArgMaxV


def AcaPlus(sig, tau, f_BEM, err, kmax=20) :

    m, n = len(sig), len(tau)
    CompCoef = 0
    CompCoefSafe = 0

    M_c = np.array([[None for j in range(n)] for i in range(m)])
    R_k = np.zeros((m,n))

    kmax_r = min(min(n,m), kmax)
    A_r, B_r = np.zeros((m,kmax_r)), np.zeros((n,kmax_r))
    
    ## initialization of j_ref, i_ref and c_ref, l_ref
    j_ref = 0 
    for i in range(m):
        M_c[i][j_ref] = f_BEM(sig[i], tau[j_ref] )
    c_ref = M_c[:,j_ref]

    i_ref = np.argmin(np.abs(c_ref))
    for j in range(n):
        if M_c[i_ref][j] == None :
            M_c[i_ref][j] = f_BEM(sig[i_ref], tau[j] )
    l_ref = M_c[i_ref,:]
    CompCoef += m+n-1

    ## initialization of Pc, Pl list , k rank, error, and test condition
    Pc, Pl = [], []
    k = 0
    r2_k = 0
    TestContinue = True
    while TestContinue : ## start of loop

        c_ref = c_ref -  R_k[:,j_ref]
        l_ref = l_ref -  R_k[i_ref,:]

        d_c_s, i_s = ArgMaxV(c_ref, Pl) 
        d_l_s, j_s = ArgMaxV(l_ref, Pc) 

        if np.abs(d_c_s) < 10e-10 and np.abs(d_l_s) < 10e-10 : ## the reference vectors are both equal to zero
            return(R_k, k, CompCoef, CompCoefSafe) #, A_r, B_r)

        if np.abs(d_c_s) > np.abs(d_l_s) : # selection of i_s

            for j in range(n) :
                if M_c[i_s][j] == None :
                    M_c[i_s][j] = f_BEM( sig[i_s], tau[j] ) 
                    CompCoef += 1
            b_k = M_c[i_s,:] - R_k[i_s,:] 
            j_s = np.argmax( np.abs(b_k) )
            d = b_k[j_s]

            for i in range(m) :
                if M_c[i][j_s] == None :
                    M_c[i][j_s] = f_BEM( sig[i], tau[j_s] )
                    CompCoef += 1
            a_k = M_c[:,j_s] - R_k[:,j_s] 
            a_k = a_k/d
        
        else : # selection of j_s, new i_s

            for i in range(m) :
                if M_c[i][j_s] == None :
                    M_c[i][j_s] = f_BEM( sig[i], tau[j_s] ) 
                    CompCoef += 1
            a_k = M_c[:,j_s] - R_k[:,j_s]
            i_s = np.argmax( np.abs(a_k) )
            d = a_k[i_s]

            for j in range(n) :
                if M_c[i_s][j] == None :
                    M_c[i_s][j] = f_BEM( sig[i_s], tau[j] )
                    CompCoef += 1
            b_k = M_c[i_s,:] - R_k[i_s,:] 
            b_k = b_k/d
        
        ## end of the a_k and b_k calculation
        # saving the row and column explored
        Pl.append( i_s )
        Pc.append( j_s )

        # iterative computation of Rk
        k += 1
        R_k = R_k + np.array([[a_k[i]*b_k[j] for j in range(n)] for i in range(m)])

        ## residual error
        r2_k_k1 = (lin.norm(a_k)*lin.norm(b_k))**2
        r2_k += 2*np.dot(Residual(A_r, a_k), np.transpose(Residual(B_r, b_k))) + r2_k_k1
        if r2_k_k1 <= err*r2_k or k == kmax_r:
            TestContinue = False 
        
        ## broadcast value in A_r, B_r matrix 
        A_r[:,k-1] = a_k
        B_r[:,k-1] = b_k 

        ## all row and column seen
        if len(Pc) == n or len(Pl) == m :
            return( R_k, k, CompCoef, CompCoefSafe) #, A_r, B_r )
        
        ## actualisation of i_ref and j_ref
        if i_s == i_ref or j_s == j_ref : # change of reference row or column
            if i_s == i_ref and j_s == j_ref : # both
                j_ref = ArgMin(n, Pc)
                for i in range(m) :
                    if M_c[i][j_ref] == None :
                        M_c[i][j_ref] = f_BEM( sig[i], tau[j_ref] ) 
                        CompCoef += 1
                c_ref = M_c[:,j_ref]
                i_ref = np.argmin(np.abs(c_ref))

                for j in range(n) :
                    if M_c[i_ref][j] == None :
                        M_c[i_ref][j] = f_BEM( sig[i_ref], tau[j] ) 
                        CompCoef += 1
                l_ref = M_c[i_ref,:]

            else :
                if j_s == j_ref : # new reference column
                    j_ref = ArgMin(n, Pc) # at random * Grasedyck 2005
                    for i in range(m) :
                        if M_c[i][j_ref] == None :
                            M_c[i][j_ref] = f_BEM( sig[i], tau[j_ref] )
                            CompCoef += 1
                    c_ref = M_c[:,j_ref]
                else : # new row reference
                    i_ref = ArgMin(m, Pl)
                    for j in range(n) :
                        if M_c[i_ref][j] == None :
                            M_c[i_ref][j] = f_BEM( sig[i_ref], tau[j] )
                            CompCoef += 1
                    l_ref = M_c[i_ref,:]
        ## end of the actualization i_ref and j_ref
        
    CompCoefSafe += k*(m+n)

    return(R_k, k, CompCoef, CompCoefSafe)
