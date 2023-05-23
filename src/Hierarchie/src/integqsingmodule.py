
import numpy as np 


from .optim.integtoolsmodule import custom_norm, custom_x_array_shape
from .gausspt.gaussmodule import gauss_tri

PI_INV = 1./(4*np.pi)


class integ_quasi_sing( object ) :


    def __init__( self  ) :

        """
            constructor for the quasi singular integration
                __init__( self )
                save the inputs as local object 
                reshape the wieghts array for the specific integration function
        """

        C_tri, W_tri = gauss_tri( )
        self.C_quasi = C_tri[1:]
        self.W_quasi = W_tri[1:].reshape(3,19,1)


    def f_quasi_sing_integ( self, J, mesh, X0, Ng ) :

        """
            function to compute the quasi singular integral
                f_quasi_sing_integ( self, J, mesh, X0, Ng )

                -- input : J the area of the element
                    - mesh the matrix of nodal coordinate 
                    - X0 the source point
                    - Ng the index of severity 
                
                -- with Ng the integration is given by a 
                
        """

        X_sing = custom_x_array_shape( self.C_quasi[Ng], mesh ) 
        f = J * PI_INV * np.dot( custom_norm( X0 - X_sing ), self.W_quasi[Ng] )[0]
        del X_sing 

        return( f )
    