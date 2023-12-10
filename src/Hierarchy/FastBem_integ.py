
import numpy as np 

import sys 
sys.path.append( '/home/pbeguin/Bureau/BEM_module_v3/Hierarchie')

## functions in optim
try :
    from .src.optim.normmodule import norm_computer_optim, is_computer_optim
    # from .src.gausspt.gaussmodule import gauss_tri
    from .src.integ6module import integ_6pt
    from .src.integsingmodule import integ_sing
    from .src.integqsingmodule import integ_quasi_sing
    from .src.integpartmodule import integ_partt
except :
    from src.optim.normmodule import norm_computer_optim, is_computer_optim
    # from gausspt.gaussmodule import gauss_tri
    from src.integ6module import integ_6pt
    from src.integsingmodule import integ_sing
    from src.integqsingmodule import integ_quasi_sing
    from src.integpartmodule import integ_partt


def load_cluster_(cluster) :
    if type(cluster) == dict :
        if 'mesh' in cluster.keys :
            return( cluster["mesh"], cluster["mesh_size"], cluster["X0_mesh"] )
        else :
            return( cluster["mesh_elem"], cluster["mesh_size"], cluster["X0_mesh"] )
    try :
        return( cluster.mesh, cluster.mesh_size, cluster.X0_mesh )
    except AttributeError : 
        return( cluster.mesh_elem, cluster.mesh_size, cluster.X0_mesh )
    
def x_gap(X, Y) :
    return norm_computer_optim( X-Y )



class Bem_integ( integ_6pt, integ_sing, integ_quasi_sing, integ_partt ) :


    def __init__(self, X0_mesh1, Cluster2, sing=False) :
        
        ## input
        self.sing = sing
        self.X0_mesh1 = X0_mesh1
        self.mesh, self.mesh_size2, self.X0_mesh2 = load_cluster_( Cluster2 )


        ## all integration modules
        integ_6pt.__init__( self, self.mesh, self.X0_mesh1 )
        integ_sing.__init__( self )
        integ_quasi_sing.__init__( self )
        integ_partt.__init__( self )
        
        ## singular integration
        self.Dmax2 = np.max( self.mesh_size2 )
        self.dmin2 = 2.37 * self.Dmax2

        # # vectorisation of the integration function
        # self.F_integ_sing_vect = np.vectorize(self.F_integ_sing)
        # self.F_integ_nsing_vect = np.vectorize(self.F_integ_nsing)
    


    def F_integ_sing(self, i, j) : 
        
        if i == j :
            return( self.f_sing_integ( self.J_mesh[j], self.mesh[j], self.X0_mesh1[i] ) )

        d_ij = x_gap(self.X0_mesh2[i], self.X0_mesh1[j])
        
        if d_ij >= self.dmin2 :
            return( self.f_6pt_integ( i, j ) )

        IS = is_computer_optim(d_ij, self.mesh_size2[j] )
        if IS == 1 :
            return( self.f_6pt_integ( i, j ) )
        elif IS < 5 :
           return( self.f_quasi_sing_integ(self.J_mesh[j], self.mesh[j], self.X0_mesh1[i], IS-2 ) )
        
        return( self.f_partt_integ(self.J_mesh[j], self.mesh[j], self.X0_mesh1[i]) )



    def F_integ_nsing(self, i, j) : 

        d_ij = x_gap(self.X0_mesh2[i], self.X0_mesh1[j])
        
        if d_ij >= self.dmin2 :
            return( self.f_6pt_integ( i, j ) )

        IS = is_computer_optim(d_ij, self.mesh_size2[j] )
        if IS == 1 :
            return( self.f_6pt_integ( i, j ) )
        elif IS < 5 :
           return( self.f_quasi_sing_integ(self.J_mesh[j], self.mesh[j], self.X0_mesh1[i], IS-2 ) )

        return( self.f_partt_integ(self.J_mesh[j], self.mesh[j], self.X0_mesh2[i]) )

