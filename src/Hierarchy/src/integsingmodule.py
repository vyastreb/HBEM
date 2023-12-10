
import numpy as np 
# import numpy.linalg as lin

from .optim.integtoolsmodule import custom_norm, custom_x_array_shape


## W_sing sum = 1
W_seg = np.array([0.08566225, 
                   0.18038079, 
                   0.23395697, 
                   0.23395697, 
                   0.18038079,
                   0.08566225]).reshape(1,6)

## C_sing on [0, 1]
C_seg = np.array([0.03376524, 
                    0.16939531, 
                    0.38069041, 
                    0.61930959, 
                    0.83060469,
                    0.96623476]).reshape(1,6)


PI_INV = 1./(4*np.pi)


### triangular separation parameter 
h = np.array( [1/3, np.sqrt(2)/6, 1/3] )
alpha1 = np.array( [-3*np.pi/4, -(np.pi)/2 + np.arctan(2), (np.pi)-np.arctan(2)] )

alpham = np.array([np.pi/4, 
                   (np.pi)/4 + np.arctan(1/2), 
                   (np.pi)/2 - np.arctan(1/2)])

dthe = np.array([(np.pi)/4 + np.arctan(2), 
                 (np.pi)/2 + 2*np.arctan(1/2), 
                 (np.pi)/4 + np.arctan(2)])

L = np.array([ [[np.cos(alpha1[k]), -np.sin(alpha1[k])],
            [np.sin(alpha1[k]),np.cos(alpha1[k])]] for k in range(3)] )


class integ_sing( object ) :


    def __init__( self ) :

        """
            constructor for the singular integration 
                -- procedure : initialisation of C_sing, and W_sing

                -- C_sing the relative coordinate in the triangle of reference 
                    nested around the centroid point (1/3, 1/3)

                -- W_sing the corresponding modified weights 
                    sum(W_sing) = 1

        """

        self.C_W_sing_init( )

    def C_W_sing_init( self ) :

        """ 
            function which initiate the array C_sing and W_sing

            -- C_sing is shape of (108, 2)
            -- W_sing is shape of (108, 1)

            -- those are specially design for the function of integration 
                f_sing_integ
        """

        C_sing = np.empty((3,6,6,2))
        W_sing = np.empty((3,6,6))
        
        the_mat = np.dot( dthe.reshape(3,1), C_seg )
        dthe_mat = np.dot( dthe.reshape(3,1), W_seg )

        R_mat = np.dot( h.reshape(3,1), np.ones((1,6)) ) / \
            np.cos( the_mat - np.dot( alpham.reshape(3,1), np.ones((1,6)) ) )
        
        for k in range(3) :

            for i in range(6) :

                r = R_mat[k][i] * C_seg
                dr = R_mat[k][i] * W_seg
                    
                e = np.dot(np.array([[np.cos( the_mat[k][i] )], [np.sin( the_mat[k][i] )]]), 
                           r.reshape(1,6) )
                
                n = np.dot( L[k], e ) + np.array([[1/3], [1/3]])
                    
                C_sing[k][i] = n.T
                W_sing[k][i] = r * dr * dthe_mat[k][i]

        self.C_sing = C_sing.reshape(108, 2)
        self.W_sing = W_sing.reshape(108, 1)

        del C_sing, W_sing 
        del the_mat, dthe_mat, R_mat 


    
    def f_sing_integ( self, J, mesh, X0 ) :

        """
            function which return the integration result of the kernel function on elem
                f_sing_sing( J, mesh, X0 )

                input : J is the area of the element , result of the Jacobian computation
                    mesh is the nodal coordinate matrix 
                    X0 the coordinate of the source point

                the integration points are first computed (denoted by X_sing) 
                    on the element (mesh) by using C_sing 

                the kernel function is computed as an array using custom_norm
                the result is given by matrix product procedure
                
        """

        X_sing = custom_x_array_shape( self.C_sing, mesh ) 
        f = J * PI_INV * np.dot( custom_norm( X0 - X_sing ), self.W_sing )[0]
        del X_sing 

        return( f )
    