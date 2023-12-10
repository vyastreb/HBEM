

import numpy as np 
import numpy.linalg as lin 

try :
    from .optim.normmodule import norm_computer_optim

except :
    from optim.normmodule import norm_computer_optim
    

err_S = 10e-10

S = np.array([[0,1], [1,2], [2,0]])
Sc = np.array([1, 2, 0])


T1 = { 'h':np.array([1]),
        'd_the':np.array([(np.pi)/4]),
        'alpha':np.array([3*(np.pi)/4]),
        'L':np.array([[[-np.sqrt(2)/2,-np.sqrt(2)/2],[np.sqrt(2)/2, -np.sqrt(2)/2]]]) }


T2 = { 'h':np.array([1]) ,
       'd_the':np.array([np.pi/4]),
       'alpha':np.array([-(np.pi)/2]),
       'L':np.array([[[0,1], [-1,0]]]) }


T3 = { 'h':np.array([np.sqrt(2)/2]),
        'd_the':np.array([(np.pi)/2]),
        'alpha':np.array([(np.pi)/4]),
        'L':np.array([[[1,0], [0,1]]]) }


def x_gap(X, Y) :
    return norm_computer_optim( X-Y )


### functions designed for integpartmodule
def xs_segment_def( elem, X_) :
    Xv = (elem[1] - elem[0]) / x_gap(elem[0], elem[1])
    Bh = np.dot( (X_ - elem[0]).T, Xv )
    return( elem[0] + Bh * Xv )


def xs_test_in_segment( elem, Xs) :

    Dseg = x_gap( elem[0], elem[1])
    D1 = x_gap( elem[0], Xs )
    D2 = x_gap( elem[1], Xs )
    
    if (D1 + D2 - Dseg) <= err_S :
        return( True )
    return( False )
    

def xs_out_segment( elem, Xs) :
    i = np.argmin( lin.norm( elem - Xs  , axis=1 ) )
    return( elem[i] )
    

def xs_def( elem, X_int) :

    Xs_seg = np.empty((3, 2))
    # Dseg = lin.norm( elem - elem[Sc], axis=1 )
    # print( Dseg )

    for i in range(3):

        Xsi = xs_segment_def(elem[S[i]], X_int)
        # print( Xsi )

        if xs_test_in_segment( elem[S[i]], Xsi ) :
            Xs_seg[i] = Xsi
        else :
            Xs_seg[i] = xs_out_segment( elem[S[i]], Xsi )
    
    # print( Xs_seg )
    ind = np.argmin( lin.norm( Xs_seg - X_int, axis=1 ) )
    Xs = Xs_seg[ind]
    
    ind_segment_list = []
    for i in range(3) :
        if xs_test_in_segment(elem[S[i]], Xs) :
            ind_segment_list.append( i )
    
    return( Xs, ind_segment_list)



def ns_def_lin( elem, Xs, i ) :

    xi, xi1 = elem[S[i,0]][0], elem[S[i,1]][0]
    yi, yi1 = elem[S[i,0]][1], elem[S[i,1]][1]
    Ns = np.array([0., 0.])

    if xi != xi1 :
        if i == 0 :
            Ns[0] = (xi - Xs[0]) / (xi - xi1)
        elif i == 1 :
            Ns[0] = (xi - Xs[0]) / (xi - xi1)
            Ns[1] = 1 - Ns[0]
        else :
            Ns[1] = (xi - Xs[0]) / (xi - xi1)
    else :
        if i == 0 :
            Ns[0] = (yi - Xs[1]) / (yi - yi1)
        elif i == 1 :
            Ns[0] = (yi - Xs[1]) / (yi - yi1)
            Ns[1] = 1 - Ns[0]
        else :
            Ns[1] = (yi - Xs[1]) / (yi - yi1)
    
    return( Ns )


def Delta_j( n_b, ind_seg)  :

    n1, n2 = n_b[0], n_b[1]

    if ind_seg == [0] :

        dthe2 = np.arctan(1/n1)
        dthe1 = (np.pi) - dthe2

        h1 = (1-n1)*np.sqrt(2)/2
        
        h_list = [h1, n1]
        d_the_list = [dthe1, dthe2]

        alpha_list = [(np.pi)/4, dthe2]

        L1 = [[1, 0],[0, 1]]
        L2 = [[np.cos(dthe1), -np.sin(dthe1)],[np.sin(dthe1), np.cos(dthe1)]]
        L_list = [L1, L2]

        return( np.array(L_list), np.array(d_the_list), np.array(alpha_list), np.array(h_list) )
        
    if ind_seg == [1] :

        h_list = [n2, n1]

        a1 = np.arctan(n1/n2)
        alpha_list = [a1, (np.pi)/4]

        al1 = -((np.pi)/2 + a1) 
        al2 = 3*(np.pi)/4

        L1 = [[np.cos(al1), -np.sin(al1)],[np.sin(al1), np.cos(al1)]]
        L2 = [[np.cos(al2), -np.sin(al2)],[np.sin(al2), np.cos(al2)]]
        L_list = [L1, L2]

        d_the_list = [a1 + (np.pi)/4, 3*(np.pi)/4 - a1]

        return( np.array(L_list), np.array(d_the_list), np.array(alpha_list), np.array(h_list) )
    
    if ind_seg == [2] :

        h_list = [n2, (1-n2)*np.sqrt(2)/2]

        dthe1 = np.arctan(1/n2)
        dthe2 = (np.pi) - dthe1
        d_the_list = [dthe1, dthe2]
        
        a1 = 0
        a2 = dthe2 - (np.pi/4)
        alpha_list = [a1, a2]

        L1 = [[0, 1],[-1, 0]]
        L2 = [[np.cos((np.pi)/2-dthe1), np.sin((np.pi)/2-dthe1)],[-np.sin((np.pi)/2-dthe1), np.cos((np.pi)/2-dthe1)]]
        L_list = [L1, L2]

        return( np.array(L_list), np.array(d_the_list), np.array(alpha_list), np.array(h_list) )
    
    if ind_seg == [0, 1]  :
        return( T1['L'], T1['d_the'], T1['alpha'], T1['h'] )

    if ind_seg == [1, 2] :
        return( T2['L'], T2['d_the'], T2['alpha'], T2['h'] )
    
    if ind_seg == [0, 2] :
        return( T3['L'], T3['d_the'], T3['alpha'], T3['h'] )

