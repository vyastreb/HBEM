
import numpy as np 
import numpy.linalg as lin

import os 
import sys 
import pathlib as pl 
sys.path.append( '/home/pbeguin/Bureau/BEM_module_v3/Hierarchie/src/optim')

try :
    from .optim.integtoolsmodule import custom_norm, custom_x_array_shape
except :
    from optim.integtoolsmodule import custom_norm, custom_x_array_shape


try :
    from .gausspt.gaussmodule import gauss_quad
    from .partttools import xs_def, ns_def_lin, Delta_j
except :
    from gausspt.gaussmodule import gauss_quad
    from partttools import xs_def, ns_def_lin, Delta_j


PI_INV = 1./(4*np.pi)


class integ_partt( object ) :


    def __init__( self, ) :

        ## gauss point Ng = 4
        C_part_, W_part_ = gauss_quad( ) 
        self.C_part = 0.5 * (1 + C_part_[2][:4]).reshape(1,4)
        self.W_part = 0.5 * (W_part_[2][:4]).reshape(1,4)


    
    def f_partt_integ( self, J, elem, X0 ) :

        ## triangle partition
        Xs, ind_segment = xs_def(elem, X0 )
        ns_ = ns_def_lin(elem, Xs, ind_segment[0])
        L, dthe, alpha, h = Delta_j(ns_, ind_segment)
        ns = ns_.reshape((2,1))

        ## gauss point 
        N = len(L)
        # print( N )

        C_inter = np.empty((N,4,4,2))
        W_inter = np.empty((N,4,4))
        
        the_mat = np.dot( dthe.reshape(N,1), self.C_part )
        dthe_mat = np.dot( dthe.reshape(N,1), self.W_part )

        R_mat = np.dot( h.reshape(N,1), np.ones((1,4)) ) / \
            np.cos( the_mat - np.dot( alpha.reshape(N,1), np.ones((1,4)) ) )
        
        for k in range(N) :

            for i in range(4) :

                r = R_mat[k][i] * self.C_part
                dr = R_mat[k][i] * self.W_part
                
                e = np.dot(np.array([[np.cos( the_mat[k][i] )], [np.sin( the_mat[k][i] )]]), 
                           r.reshape(1,4) )
                
                n = np.dot( L[k], e ) + ns
                    
                C_inter[k][i] = n.T
                W_inter[k][i] = 2 * r * dr * dthe_mat[k][i]

        C_inter = C_inter.reshape(N*16, 2)
        W_inter = W_inter.reshape(N*16, 1)

        ## final summation
        X_sing = custom_x_array_shape( C_inter, elem ) 
        f = J * PI_INV * np.dot( custom_norm( X0 - X_sing ), W_inter )[0]
        del C_inter, W_inter, X_sing
        
        return( f )