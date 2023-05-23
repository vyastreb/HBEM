# cimport numpy as cnp 

import numpy as np 
import numpy.linalg as lin

from .optim.integtoolsmodule import custom_norm, custom_x_array_shape #, custom_x_shape
from .gausspt.gaussmodule import gauss_6pt

## weights 6 pt
W = np.array( [0.111690794839005, 
               0.111690794839005, 
               0.111690794839005, 
               0.054975871827661, 
               0.054975871827661, 
               0.054975871827661] ).reshape(1,6)

## coords 6 pt
C = np.array( [ [0.445948490915965, 0.445948490915965], 
                [0.10810301816807, 0.445948490915965], 
                [0.445948490915965, 0.10810301816807], 
                [0.091576213509771, 0.091576213509771], 
                [0.816847572980458, 0.091576213509771], 
                [0.091576213509771, 0.816847572980458] ] )

## Jacobian matrxi of shape functions
J = np.array( [ [-1., 1., 0.], 
                [-1., 0., 1.] ] )


PI_INV = 1./(4*np.pi)


## class
class integ_6pt( object ) :

    def __init__( self, mesh, X0_mesh ) :

        """ 
            Constructor -- computation of the integrating coordinate point
        """

        # C_6pt_, W_6pt_ = gauss_6pt( )
        # self.C_6pt, self.W_6pt = C_6pt_, W_6pt_.reshape((1,6))

        self.X0_mesh = X0_mesh
        self.J_mesh_init( mesh )
        self.W_6pt_init( )
        self.X_6pt_init( mesh )


    def J_mesh_init( self, mesh ) :

        """
            Function to initiate the Jacobian on all element
                -- calling up the matrix J 
                -- J takes advantage of the linear trangular property for the constant interpolation
                -- J is made of constant coefficient , non dependent on the nodal coordinates

                -- J_e = | det( G_6pt . mesh_e ) | for an nodal matrix mesh_e

                -- resulting of the creation of J_mesh vector lenght of n (number of element)
                -- used in W_6pt_mesh computation , the modified weights for all elements
        """

        self.J_mesh = np.array( [ np.abs( lin.det( np.dot( J, mesh_e ) ) ) for mesh_e in mesh ] )


    def W_6pt_init( self ) :

        """
            Function to initiate the Jacobian matrix 
                -- the modified weight matrix 
                -- with linear shape function 
                -- and triangular element
                -- result : W_6pt_mesh matrix of shape (n, 6)
        """

        self.W_6pt_mesh = np.dot( np.array( [[Je] for Je in self.J_mesh ]) , W ).reshape(self.J_mesh.shape[0], 6, 1)


    def X_6pt_init( self, mesh ) :

        """
            Function to initiate the coordinate of integration point on all element
                -- the modified weight matrix 
                -- with linear shape function 
                -- and triangular element
                -- result : X_6pt_mesh matrix of shape (n, 6, 2)
                -- using optimize function custom_x_array_shape
        """

        self.X_6pt_mesh = np.array( [custom_x_array_shape( C, mesh_e ) for mesh_e in mesh ])


    def f_6pt_integ( self, i, j ) :

        """
            function for non singular integration 
                -- the source point i, and the element of integration are far apart

                -- f_{ij} = \sum_{k}^6 ( w_k / ( 4*pi*(|X_i - X_k| ) )
                -- w_k is the modified weight (W_6pt_mesh[j,k])
                -- X_i the source point (X0_mesh[i])
                -- Y_k the integrating point (X_6pt_mesh[j,k])

                -- using custom_norm

        """

        return( PI_INV * np.dot( custom_norm( self.X0_mesh[i] - self.X_6pt_mesh[j] ), self.W_6pt_mesh[j] )[0] )
        
