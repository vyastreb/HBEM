
### moduls
import numpy as np 
import numpy.linalg as lin 

## imports tool function
# from BEM_modul_v2.Core.tools import x_gap

def x_gap(X, Y) :
     return(lin.norm(X - Y))

from .tools import Is


### class declaration
class Cluster :
    
    """
    Class Cluster 
    """
    
    def __init__(self, mesh, Sleaf, opt='med') :
        
        """
        Constructor for mesh, Sleaf, and Ne (number of element)
        """

        ## input
        self.mesh = mesh.mesh_elem 
        self.mesh_size = mesh.mesh_size
        self.X0_mesh = mesh.X0_mesh
        self.Ne = mesh.Ne
        self.Sleaf = Sleaf

        ## type of split
        self.opt = opt

        self.S_elem0 = np.arange(self.Ne)
        Size0 = self.SizeS_def(self.mesh)


        ## list object to build with the mesh
        self.STotElem = []
        self.STot = []
        self.SLevel = []
        self.SizeLevel = []

        ## variable of relative position and level
        self.Nlevel = 0
        self.IndS = 0

        ## definition of the 
        self.init_cluster_tree(self.X0_mesh, self.mesh, self.S_elem0, Size0)
        

    
    def SizeS_def(self, S_mesh) :

        """
        Function to return the box size of the element set
        --- S_mesh input, array of shape (Ne, 3, 2) for plane triangular element

        -- first ravel all the X, Y nodal coordinates
        -- take the length along X, and Y axis, (respectively, d1 and d2) 
        -- and return the size (diagonal) of the box
        """
        X_mesh = np.array(S_mesh)[:,:,0].reshape((S_mesh.shape[0]*S_mesh.shape[1],1))
        Y_mesh = np.array(S_mesh)[:,:,1].reshape((S_mesh.shape[0]*S_mesh.shape[1],1))
        d1 = np.max(X_mesh) - np.min(X_mesh)
        d2 = np.max(Y_mesh) - np.min(Y_mesh)
        del X_mesh, Y_mesh
        return( np.sqrt(d1*d1 + d2*d2) )


    def Split(self, S_X0, S_mesh, S_elem) : # code function for splitng a domain accounting on median algorithm

        """ 
        Function to split an element set in two
            -- split along either x, y axis , def is 
            -- find the median along this axis if opt = 'med'
                -- geom criteria for opt = 'geo'

            return S, S_elem, S_size for the two sets
        """

        ### def is and the median
        i_s = Is(S_mesh)
        if self.opt == 'med' :
            x_is_seq = np.median( S_X0[:,i_s] )
        elif self.opt in ['geo'] :
            x_is_seq = 0.5 * (np.max( S_X0[:,i_s] ) + np.min( S_X0[:,i_s] ) )
        

        ### split and construction of the sets
        S1_X0_, S2_X0_ = [], []
        S1_mesh_, S2_mesh_ = [], []
        S1_elem, S2_elem = [], []

        for k, Sk in enumerate(S_X0) :
            if Sk[i_s] <= x_is_seq :
                S1_X0_.append( Sk.tolist() )
                S1_mesh_.append( S_mesh[k].tolist() )
                S1_elem.append( S_elem[k] )
            else: 
                S2_X0_.append( Sk.tolist() )
                S2_mesh_.append( S_mesh[k].tolist() )
                S2_elem.append( S_elem[k] )
        
        S1_mesh = np.array( S1_mesh_ )
        S2_mesh = np.array( S2_mesh_ )
        S1_X0, S2_X0 = np.array( S1_X0_ ), np.array( S2_X0_ )
        del S1_mesh_, S2_mesh_, S1_X0_, S2_X0_

        ### sizes for the two sets
        S_ex_1 = self.SizeS_def( S1_mesh )
        S_ex_2 = self.SizeS_def( S2_mesh )

        return( S1_X0, S2_X0, S1_mesh, S2_mesh, S1_elem, S2_elem , S_ex_1, S_ex_2)

    
    def SplitGeoMed(self, S_X0, S_mesh, S_elem) : 

        """ 
        Function to split an element set in two
            -- split along either x, y axis , def is 
            -- find the median along this axis if opt = 'med'
                -- geom criteria for opt = 'geo'

            return S, S_elem, S_size for the two sets
        """

        ### def is and the median
        i_s = Is(S_mesh)
        x_is_seq = np.mean( S_X0[:,i_s] )
        

        ### split and construction of the sets
        S1_X0, S2_X0 = [], []
        S1_mesh, S2_mesh = [], []
        S1_elem, S2_elem = [], []

        for k, Sk in enumerate(S_X0) :
            if Sk[i_s] <= x_is_seq :
                S1_X0.append( Sk.tolist() )
                S1_mesh.append( S_mesh[k].tolist() )
                S1_elem.append( S_elem[k] )
            else: 
                S2_X0.append( Sk.tolist() )
                S2_mesh.append( S_mesh[k].tolist() )
                S2_elem.append( S_elem[k] )
        
        ### shape optimization
        test = False
        if np.abs( len(S1_X0) - len(S2_X0) ) > 0.1 * max( len(S1_X0), len(S2_X0) ) :
            N1, N2 = len( S1_X0 ), len( S2_X0 )

            x0_ref1 = ( np.max( np.array( S1_mesh )[:,:,0] ) - np.min( np.array( S1_mesh )[:,:,0] ) ) / 2
            y0_ref1 = ( np.max( np.array( S1_mesh )[:,:,1] ) - np.min( np.array( S1_mesh )[:,:,1] ) ) / 2

            x0_ref2 = ( np.max( np.array( S2_mesh )[:,:,0] ) - np.min( np.array( S2_mesh )[:,:,0] ) ) / 2
            y0_ref2 = ( np.max( np.array( S2_mesh )[:,:,1] ) - np.min( np.array( S2_mesh )[:,:,1] ) ) / 2

            X0_ref1, X0_ref2 = np.array( [x0_ref1, y0_ref1] ), np.array( [x0_ref2, y0_ref2])
            size1, size2 = self.SizeS_def( np.array( S1_mesh ) ), self.SizeS_def( np.array( S2_mesh ) )

            test = True

        while test :

            if N1 < N2 :
                ind = np.argmin( np.array( [ (X0_ref1[0] - X2[0])**2 + (X0_ref1[1] - X2[1])**2 for X2 in S2_X0 ] ) )
                S1_mesh_cp = S1_mesh.copy()
                S1_mesh_cp.append( S2_mesh[ind] )
                size_iter1 = self.SizeS_def( np.array( S1_mesh_cp ) ) #+ S2_mesh[ind] ) )

                if ( size_iter1 - size1 ) / size1 < 0.1 :
                    S1_X0.append( S2_X0[ind] )
                    S1_mesh.append( S2_mesh[ind] )
                    S1_elem.append( S2_elem[ind] )

                    S2_X0.pop(ind)
                    S2_mesh.pop(ind)
                    S2_elem.pop(ind)
                else :
                    test = False
            
            else :
                ind = np.argmin( np.array( [ (X0_ref2[0] - X1[0])**2 + (X0_ref2[1] - X1[1])**2 for X1 in S1_X0 ] ) )
                S2_mesh_cp = S2_mesh.copy()
                S2_mesh_cp.append( S1_mesh[ind] )
                size_iter2 = self.SizeS_def( np.array( S2_mesh_cp ) ) #+ S1_mesh[ind] ) )

                if ( size_iter2 - size2 ) / size2 < 0.1 :
                    S2_X0.append( S1_X0[ind] )
                    S2_mesh.append( S1_mesh[ind] )
                    S2_elem.append( S1_elem[ind] )

                    S1_X0.pop(ind)
                    S1_mesh.pop(ind)
                    S1_elem.pop(ind)
                else :
                    test = False
            
            N1, N2 = len(S1_X0), len(S2_X0)
            if np.abs( N1 - N2 ) < 3 :
                test = False
            
        ### sizes for the two sets
        S_ex_1 = self.SizeS_def(S1_mesh)
        S_ex_2 = self.SizeS_def(S2_mesh)

        return( np.array(S1_X0), np.array(S2_X0), np.array(S1_mesh), np.array(S2_mesh), S1_elem, S2_elem , S_ex_1, S_ex_2)

    

    def init_cluster_tree(self, S_X0, S_mesh, S_elem, S_ex ) :

        """
        Function to build the cluster tree
            -- recursive procedure 
            -- when the program arrive at Nlevel == Sleaf, it stops, and it increment the STotElem, etc..
            -- return the list of cluster index, 
                the first level includes all the clusters
        """

        ## new call
        self.Nlevel += 1

        if self.Nlevel == self.Sleaf :

            self.STotElem.append(S_elem)
            self.STot.append(S_X0)
            self.IndS += 1
            
            if len(self.SLevel) == self.Nlevel :
                self.SLevel[-1].append([self.IndS])
                self.SizeLevel[-1].append(S_ex)
            else :
                self.SLevel.append([[self.IndS]])
                self.SizeLevel.append([S_ex])

            return( [self.IndS] )
            
            
        else:

            ### split element set
            if self.opt == 'geo_med' :
                S1_X0, S2_X0, S1_mesh, S2_mesh, S1_elem, S2_elem, S1_ex, S2_ex = self.SplitGeoMed(S_X0, S_mesh, S_elem)
            else :
                S1_X0, S2_X0, S1_mesh, S2_mesh, S1_elem, S2_elem, S1_ex, S2_ex = self.Split(S_X0, S_mesh, S_elem)

            ### Slevel not large enough
            if self.Nlevel >= len(self.SLevel) :
                self.SLevel.append([])
                self.SizeLevel.append([])
            
            ### recursive call
            SInter1 = self.init_cluster_tree(S1_X0, S1_mesh, S1_elem, S1_ex)
            self.Nlevel -= 1
            SInter2 = self.init_cluster_tree(S2_X0, S2_mesh, S2_elem, S2_ex)


            self.Nlevel -= 1
            self.SLevel[self.Nlevel-1].append(SInter1 + SInter2)
            self.SizeLevel[self.Nlevel-1].append(S_ex)

            return( SInter1 + SInter2 )
    