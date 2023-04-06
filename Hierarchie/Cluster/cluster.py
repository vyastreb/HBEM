
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
    
    def __init__(self, mesh, Sleaf) :
        
        """
        Constructor for mesh, Sleaf, and Ne (number of element)
        """

        ## input
        self.mesh = mesh.mesh_elem 
        self.mesh_size = mesh.mesh_size
        self.X0_mesh = mesh.X0_mesh
        self.Ne = mesh.Ne
        self.Sleaf = Sleaf

        # self.Ne = len(self.mesh)
        self.S_elem0 = np.arange(self.Ne)


        ## list object to build with the mesh
        self.STotElem = []
        self.STot = []
        self.SLevel = []
        self.SizeLevel = []

        ## variable of relative position and level
        self.Nlevel = 0
        self.IndS = 0

        # ## size of the element
        # self.init_S_mesh()   
        
        ## size of the total mesh
        Size0 = self.SizeS_def(self.S_elem0)

        ## definition of the 
        self.init_cluster_tree(self.X0_mesh, self.S_elem0, Size0)
        
        

    # def init_S_mesh(self, ) :

    #     """
    #     Procedure to define the barycenter point in each element
    #         -- X0_mesh , array of (Ne,2)
    #     """
        
        
    
    
    def SizeS_def(self, S_elem) :

        """
        Function to return the box size of the element set
        """

        CoordPtS = self.mesh[S_elem]
        return(lin.norm(np.max(CoordPtS, axis=0) - np.min(CoordPtS, axis=0)))


    def SplitMedian(self, S, S_elem) : # code function for splitng a domain accounting on median algorithm

        """ 
        Function to split an element set in two
            -- split along either x, y axis , def is 
            -- find the median along this axis 

            return S, S_elem, S_size for the two sets
        """

        ### def is and the median
        i_s = Is(S)
        x_is_seq = np.median( S[:,i_s] )

        ### split and construction of the sets
        S1, S2 = [], []
        S1_elem, S2_elem = [], []

        for k, Sk in enumerate(S) :
            if Sk[i_s] <= x_is_seq :
                S1.append( Sk.tolist() )
                S1_elem.append( S_elem[k] )
            else: 
                S2.append( Sk.tolist() )
                S2_elem.append( S_elem[k] )
        
        ### sizes for the two sets
        S_ex_1 = self.SizeS_def(S1_elem)
        S_ex_2 = self.SizeS_def(S2_elem)

        return( np.array(S1), np.array(S2), S1_elem, S2_elem , S_ex_1, S_ex_2)

    
    def init_cluster_tree(self, S, S_elem , S_ex ) :

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
            self.STot.append(S)
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
            S1, S2, S1_elem, S2_elem, S1_ex, S2_ex = self.SplitMedian(S, S_elem)

            ### Slevel not large enough
            if self.Nlevel >= len(self.SLevel) :
                self.SLevel.append([])
                self.SizeLevel.append([])
            
            ### recursive call
            SInter1 = self.init_cluster_tree(S1, S1_elem, S1_ex)

            self.Nlevel -= 1
            SInter2 = self.init_cluster_tree(S2, S2_elem, S2_ex)

            self.Nlevel -= 1
            self.SLevel[self.Nlevel-1].append(SInter1 + SInter2)
            self.SizeLevel[self.Nlevel-1].append(S_ex)

            return( SInter1 + SInter2 )
    