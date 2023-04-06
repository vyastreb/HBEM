
## modules
import numpy as np
import numpy.linalg as lin
import os
import time 
import scipy.linalg as sclin

## matplotlib for plotBlockTree fucntion
import matplotlib.pyplot as plt

### functions 
from .FastBem_integ import Bem_integ
# import sys 
# sys.path.append('/home/pbeguin/Bureau/BEM_modul')
# from BEM_integ_tri_lin_H import BEM_integ_tri_lin
from .FastBemTools import DimSLevel_def
from .AcaPlus import AcaPlus


class FastBem( Bem_integ ) : 

    """
    FastBem( cluster1, cluster2, eta, err, errb, path_save)
        -- cluster1, cluster2 the sets of element
        -- eta the admissible parameter 
        -- err, the error parameter for the ACA
        -- errb, the error parameter for the second optimization 
        -- path_save, the path folder for temporaty saving during the process
    """

    def __init__(self, Cluster1, Cluster2, eta, err, err_bis, path_save, sing=False) :
        
        """
        Constructor, Cluster1, Cluster2 for the seperate sets of elements
            -- cluster1, the elements on which the kernel is integrates
            -- cluster2, from the elements of cluster2

            -- build
            -   F_tree_adm, the H_matrix of admissibily condition
            -   Af_tree, Bf_tree, F_tree_rk, the H_matrix, obtained from ACA process
            -   Af_opti_tree, Bf_opti_tree, F_opti_tree, F_opti_adm, with the second optimization
        """

        ### input
        ## mesh
        self.sing = sing
        Bem_integ.__init__(self, Cluster1, Cluster2, self.sing)

        ## error and ACA
        self.err = err
        self.err2 = err*err
        self.err_bis = err_bis
        
        self.eta = eta
        self.path_save = path_save

        ## building the admissible matrix
        self.DimSLevel1, self.AssemSLevel1 = DimSLevel_def(self.SLevel1, self.STotElem1)
        self.DimSLevel2, self.AssemSLevel2 = DimSLevel_def(self.SLevel2, self.STotElem2)

        ## constuction of admissibility tree
        self.Nlevel = 0
        self.F_tree_adm = self.CreateBlockTree( 0, 0 )
        
        ## set the coefficient compter
        self.CompCoef = 0
        self.CompCoefSafe = 0
        self.CompCoefSafeOpti = 0
        
        ## Fast integration using ACA
        self.t_ACA = 0
        self.t_block_nsing, self.t_block_sing = 0, 0
        self.Nlevel = 0
        self.Af_tree, self.Bf_tree, self.F_tree_rk = self.CreateBlockTreeBem(0, 0)

        os.chdir(self.path_save)
        np.savez('temp_Af_Bf.npz', Af_tree = self.Af_tree, Bf_tree = self.Bf_tree, F_tree_rk = self.F_tree_rk)
        
        ## optimisation 
        self.Af_opti_tree, self.Bf_opti_tree, self.F_opti_tree_rk, self.F_opti_adm = self.Optimisation(self.Af_tree, self.Bf_tree, self.F_tree_rk, self.F_tree_adm, 0, 0)
        
        os.chdir(self.path_save)
        np.savez('temp_opti.npz', Af_opti_tree = self.Af_opti_tree, Bf_opti_tree = self.Bf_opti_tree, F_opti_tree_rk = self.F_opti_tree_rk)

        ## assemble
        # self.F_opti = self.AssembleBlockTree_AB(self.AF_opti_tree, self.BF_opti_tree, 0, 0)


    def D_def(self, pos1, pos2) :

        """
        D_def(pos1, pos2) - the distance between two sets of elements
        -- input :  pos1, pos2 
        -     relative position of set1, set2 of elements at the depth self.Nlevel
        
        -- details :
        -     distance computed as sqrt( d1*d1 + d2*d2 + d3*d3 + d4*d4 )
        -     d1, d2, d3, d4 box distances surroundings the element sets 
        """

        S1 = []
        for s_i in self.SLevel1[self.Nlevel-1][pos1] :
            S1 += self.STot1[s_i-1].tolist()
        S1 = np.array( S1 )
        
        S2 = []
        for s_i in self.SLevel2[self.Nlevel-1][pos2] :
            S2 += self.STot2[s_i-1].tolist()
        S2 = np.array( S2 )

        d1 = max(0, np.min(S1[:,0])-np.max(S2[:,0]) )
        d2 = max(0, np.min(S1[:,1])-np.max(S2[:,1]) )
        d3 = max(0, np.min(S2[:,0])-np.max(S1[:,0]) )
        d4 = max(0, np.min(S2[:,1])-np.max(S1[:,1]) )

        return( np.sqrt( np.sum( d1*d1 + d2*d2 + d3*d3 + d4*d4 ) ) )
    

    def Admissible_def(self, pos1, pos2) :

        """ 
        Admissible_def(pos1, pos2) - for the admissibility condition
        -- input : pos1, pos2 
        -     relative position of set1, set2 of elements at the depth self.Nlevel

        -- output : 1 (if min(size1, size2) <= eta * d_12 ) 
        -     0 otherwise

        -- details :
        -     d_12, the distance between clusters ( D_def(pos1, pos2) ) 
        -     size1, size2, the clusters sizes (respectively self.SizeLevel1[self.Nlevel-1][pos1], self.SizeLevel2[self.Nlevel-1][pos2])
        """

        size1, size2 = self.SizeLevel1[self.Nlevel-1][pos1], self.SizeLevel2[self.Nlevel-1][pos2]
        d_12 = self.D_def(pos1, pos2)

        if min(size1, size2) <= self.eta*d_12 :
            return(1)
        return(0)
        

    def CreateBlockTree(self, pos1, pos2) :

        """
        CreateBlockTree(pos1, pos2) - construction of the admissible tree
        --- input : pos1, pos2 
        -      relative positions of set1, set2 of elements at the depth self.Nlevel

        --- ouput : [0] if admissibility, [1] otherwise

        --- details :
        -      recursive procedure, at a deeper tree level self.Nlevel += 1 
        -      travel in the tree_rank with relative positions pos1, pos2 induced by their previous indeces
        -      and recursive assembly returning an hierarchical structure 
        """

        self.Nlevel += 1

        if self.Admissible_def(pos1, pos2)==1 :
            self.Nlevel -= 1
            return( [0] )

        elif len(self.SLevel1[self.Nlevel-1][pos1])==1 or len(self.SLevel2[self.Nlevel-1][pos2])==1 :
            self.Nlevel -= 1
            return( [1] )

        # else :
        M_r_i = np.ndarray( shape=(2,2), dtype=np.ndarray )

        M_r_i[0][0] = self.CreateBlockTree(pos1*2, pos2*2)
        M_r_i[1][0] = self.CreateBlockTree(pos1*2+1, pos2*2)
        M_r_i[0][1] = self.CreateBlockTree(pos1*2, pos2*2+1)
        M_r_i[1][1] = self.CreateBlockTree(pos1*2+1, pos2*2+1)
        self.Nlevel -= 1

        return( M_r_i )


    def SvdRk2(self, M) :

        """
        SvdRk2(M) - Svd decomposition used after ACA construction
        --- input : M block matrix 

        --- output :
        -     dot(A, diag(s[:k])), left hand reduced matrix 
        -     V[:k], right hand reduced matrix 
        -     k the reduced rank

        --- details :
        -     svd procedure inherited from numpy.linalg.svd() 
        -     the svd function provides the matrices A, V, and the vector of the singular values s
        -     first optimisation, the norm of the singular value must be lower than errb*s[0]
        -     s[0] the first singular value 
        """

        A, s, Vh = lin.svd(M.astype('float64'), full_matrices=True)
        s_selec = np.where(s >= self.err_bis*s[0])[0]
        return( A[:,s_selec]*s[s_selec], Vh[s_selec,:], [len(s_selec)] )

    
    def CreateBlockTreeBem(self, pos1, pos2) :

        """
        CreateBlockTreeBem(pos1, pos2) - integration Bem
        --- input : pos1, pos2 

        --- output : two H_matrix A, B, and the rank k 
        -     A, the left hand side H_matrix
        -     B, the right hand side H_matrix
        -       if block non-admissible, B is empty 

        --- details :
        -     if blocks admissibles, blocks computed by Aca, and Svd post traitement for the best approximation
        -     elif last scale and non-admissible, block fully integrated
        -     else, recursive call, see if the blocks are admissible at the next stage

        """

        self.Nlevel += 1

        if self.Admissible_def(pos1, pos2) == 1 : ## if blocks admissible
            
            Sig1, Tau2 = self.AssemSLevel1[self.Nlevel-1][pos1], self.AssemSLevel2[self.Nlevel-1][pos2]
            Rk, _, compt1, compt2 = AcaPlus(Sig1, Tau2, self.F_integ_nsing, self.err2)
            self.CompCoef += compt1
            self.CompCoefSafe += compt2
            self.Nlevel -= 1
            return( self.SvdRk2(Rk) )

        
        elif len(self.SLevel1[self.Nlevel-1][pos1])==1 or len(self.SLevel2[self.Nlevel-1][pos2])==1 : ## elif non admissible

            m, n = self.DimSLevel1[self.Nlevel-1][pos1], self.DimSLevel2[self.Nlevel-1][pos2]
            self.CompCoef += m*n
            self.CompCoefSafe += m*n
            self.Nlevel -= 1

            if self.sing == True :
                return( np.array( [ [ self.F_integ_sing(self.STotElem1[pos1][i], self.STotElem2[pos2][j]) 
                    for j in range(n) ] for i in range(m) ] ), [], [min(m,n)] )

            return( np.array( [ [ self.F_integ_nsing(self.STotElem1[pos1][i], self.STotElem2[pos2][j]) 
                for j in range(n) ] for i in range(m) ] ), [], [min(m,n)] )

        ## else recursive call
        A_r_i = np.ndarray( shape=(2,2), dtype=np.ndarray )
        B_r_i = np.ndarray( shape=(2,2), dtype=np.ndarray )
        Mk_r_i = np.ndarray( shape=(2,2), dtype=np.ndarray )

        A_r_i[0][0], B_r_i[0][0], Mk_r_i[0][0] = self.CreateBlockTreeBem(pos1*2, pos2*2)
        A_r_i[1][0], B_r_i[1][0], Mk_r_i[1][0] = self.CreateBlockTreeBem(pos1*2+1, pos2*2)
        A_r_i[0][1], B_r_i[0][1], Mk_r_i[0][1] = self.CreateBlockTreeBem(pos1*2, pos2*2+1)
        A_r_i[1][1], B_r_i[1][1], Mk_r_i[1][1] = self.CreateBlockTreeBem(pos1*2+1, pos2*2+1)

        self.Nlevel -= 1 
        return( A_r_i, B_r_i, Mk_r_i )
    

    def AssembleBlockTree(self, M, pos1, pos2, i=0) :

        """
        AssembleBlockTree(M, pos1, pos2) - assemble a H_matrix

        --- input : M H_matrix, pos1, pos2 
        -      pos1, pos2 relative positions in cluster tree
        -      M H_matrix with full sub block

        --- output : one matrix fully assemble

        --- details :
        -      recursive call, 
        -      pos1, pos2 used for the length of cluster sets, and broadcasting
        """

        if self.Mfullblock_test(M) :
            return( M )

        self.Nlevel += 1

        if i == 0 :
            S1, S2 = self.DimSLevel1[self.Nlevel-1][pos1], self.DimSLevel2[self.Nlevel-1][pos2]
            S1_2, S2_2 = self.DimSLevel1[self.Nlevel][pos1*2], self.DimSLevel2[self.Nlevel][pos2*2]
        if i == 1 :
            S1, S2 = self.DimSLevel2[self.Nlevel-1][pos1], self.DimSLevel3[self.Nlevel-1][pos2]
            S1_2, S2_2 = self.DimSLevel2[self.Nlevel][pos1*2], self.DimSLevel3[self.Nlevel][pos2*2]
        
        M_out = np.zeros((S1, S2))
        M_out[:S1_2,:S2_2] = self.AssembleBlockTree(M[0][0], pos1*2, pos2*2)
        M_out[:S1_2,S2_2:] = self.AssembleBlockTree(M[0][1], pos1*2, pos2*2+1)
        M_out[S1_2:,:S2_2] = self.AssembleBlockTree(M[1][0], pos1*2+1, pos2*2)
        M_out[S1_2:,S2_2:] = self.AssembleBlockTree(M[1][1], pos1*2+1, pos2*2+1)

        self.Nlevel -= 1 
        return( M_out )
    
    
    def AssembleBlockTree_AB(self, A, B, pos1, pos2) :

        """
        AssembleBlockTree(A, B, pos1, pos2) - assemble a H_matrix

        --- input : M H_matrix, pos1, pos2 
        -      pos1, pos2 relative positions in cluster tree
        -      A the right hand side H_matrix 
        -      B the left hand side H_matrix

        --- output : one matrix fully assemble

        --- details :
        -      same procedure as AssembleBlockTree
        -      with incomplete sublock for the admissible clusters,
        -      the subblock is so provided by the matrix product A.B
        """

        if (type(A[0][0]) == np.float64 ) or (type(A[0][0]) == np.float32 ) :
            if len( B ) == 0 :
                return( A ) ## sub-block non admissible
            else :
                return( np.dot( A, B ) ) ## sub_block admissible , A B product

        else :
            self.Nlevel += 1

            S1, S2 = self.DimSLevel1[self.Nlevel-1][pos1], self.DimSLevel2[self.Nlevel-1][pos2]
            S1_2, S2_2 = self.DimSLevel1[self.Nlevel][pos1*2], self.DimSLevel2[self.Nlevel][pos2*2]

            M_out = np.zeros((S1, S2))
            M_out[:S1_2,:S2_2] = self.AssembleBlockTree_AB(A[0][0], B[0][0], pos1*2, pos2*2)
            M_out[:S1_2,S2_2:] = self.AssembleBlockTree_AB(A[0][1], B[0][1], pos1*2, pos2*2+1)
            M_out[S1_2:,:S2_2] = self.AssembleBlockTree_AB(A[1][0], B[1][0], pos1*2+1, pos2*2)
            M_out[S1_2:,S2_2:] = self.AssembleBlockTree_AB(A[1][1], B[1][1], pos1*2+1, pos2*2+1)

            self.Nlevel -= 1 
            return( M_out )


    def AssembleBlockTree_Frk(self, F_rk, pos1, pos2, rk) :

        """
        AssembleBlockTree_Frk(A, B, pos1, pos2) - assemble a rank H_matrix, or the admissible tree

        --- input : F_rk, pos1, pos2 
        -      F_rk, rank or admissible H_matrix
        -      pos1, pos2 relative positions in cluster tree
        -      rk = true, for rank matrix, rk = 0 for admissible tree

        --- output : one matrix fully assemble

        --- details :
        -      same procedure as AssembleBlockTree
        """

        if (type(F_rk[0]) == float) or (type(F_rk[0]) == int) :
            m, n = self.DimSLevel1[self.Nlevel][pos1], self.DimSLevel2[self.Nlevel][pos2]

            if rk :
                return( F_rk[0]/min(m,n) * np.ones((m, n)) ) 

            return( F_rk[0]*np.ones((m, n)) )

        else :

            self.Nlevel += 1

            S1, S2 = self.DimSLevel1[self.Nlevel-1][pos1], self.DimSLevel2[self.Nlevel-1][pos2]
            S1_2, S2_2 = self.DimSLevel1[self.Nlevel][pos1*2], self.DimSLevel2[self.Nlevel][pos2*2]
            
            F_rk_out = np.zeros((S1, S2))
            F_rk_out[:S1_2,:S2_2] = self.AssembleBlockTree_Frk(F_rk[0][0], pos1*2, pos2*2, rk)
            F_rk_out[:S1_2,S2_2:] = self.AssembleBlockTree_Frk(F_rk[0][1], pos1*2, pos2*2+1, rk)
            F_rk_out[S1_2:,:S2_2] = self.AssembleBlockTree_Frk(F_rk[1][0], pos1*2+1, pos2*2, rk)
            F_rk_out[S1_2:,S2_2:] = self.AssembleBlockTree_Frk(F_rk[1][1], pos1*2+1, pos2*2+1, rk)

            self.Nlevel -= 1 
            return( F_rk_out ) 

    
    def SubAssembly(self, M1, M2, M3, M4) :

        """
        SubAssembly(M1, M2, M3, M4) - function to create a new array [[M1, M2], [M3, M4]]
            - used for the output of the optimisation process
        """

        M_out = np.ndarray(shape=(2,2), dtype=np.ndarray)
        M_out[0][0] = M1
        M_out[0][1] = M2
        M_out[1][0] = M3
        M_out[1][1] = M4
        return( M_out )


    def Optimisation(self, A, B, M_rk, M_adm, pos1, pos2 ) :

        """ 
        Optimisation(A, B, M_rk, M_adm, pos1, pos2) - optimisation process - second svd
        --- input : A, B, M_rk, M_adm, pos1, pos2 
        -      A, B, H_matrices (left and right hand side)
        -      M_rk , rank H_matrix
        -      M_adm, admissible H_matrix
        -      pos1, pos2, relative position in the cluster tree

        --- output : A_out, B_out, M_rk_out, adm 
        -      A_out, B_out, optimised H_matrices
        -      M_rk_out, optimised rank H_matrix
        -      adm = True if a new optimisation is made , False otherwise

        --- details :
        -     a new optimisation is made, if the all the subblock are admissible
        -     and if the optimisation conduct to a better rank
        """

        if type( A[0][0] ) == np.float64 :
            if M_adm[0] == 0 :
                return( np.array(A, dtype='float32'), np.array(B, dtype='float32'), M_rk, True) ## lowest block admissible
            m, n = A.shape
            self.CompCoefSafeOpti += m*n   
            return( np.array(A, dtype='float32'), [], M_rk, False)

        else :

            self.Nlevel += 1

            A11, B11, M11_rk, M11_adm = self.Optimisation(A[0][0], B[0][0], M_rk[0][0], M_adm[0][0], pos1*2, pos2*2)
            A12, B12, M12_rk, M12_adm = self.Optimisation(A[0][1], B[0][1], M_rk[0][1], M_adm[0][1], pos1*2, pos2*2+1)
            A21, B21, M21_rk, M21_adm = self.Optimisation(A[1][0], B[1][0], M_rk[1][0], M_adm[1][0], pos1*2+1, pos2*2)
            A22, B22, M22_rk, M22_adm = self.Optimisation(A[1][1], B[1][1], M_rk[1][1], M_adm[1][1], pos1*2+1, pos2*2+1)
            
            self.Nlevel -= 1

            if all([M11_adm, M12_adm, M21_adm, M22_adm]) == False : ## at least one son is non-admissible , no-coarsening
                if M11_adm :
                    m, n = self.DimSLevel1[self.Nlevel+1][pos1*2], self.DimSLevel2[self.Nlevel+1][pos2*2]
                    self.CompCoefSafeOpti += M11_rk[0]*max(m, n)
                if M21_adm :
                    m, n = self.DimSLevel1[self.Nlevel+1][pos1*2+1], self.DimSLevel2[self.Nlevel+1][pos2*2]
                    self.CompCoefSafeOpti += M21_rk[0]*max(m, n)
                if M12_adm :
                    m, n = self.DimSLevel1[self.Nlevel+1][pos1*2], self.DimSLevel2[self.Nlevel+1][pos2*2+1]
                    self.CompCoefSafeOpti += M12_rk[0]*max(m, n)
                if M22_adm :
                    m, n = self.DimSLevel1[self.Nlevel+1][pos1*2+1], self.DimSLevel2[self.Nlevel+1][pos2*2+1]
                    self.CompCoefSafeOpti += M22_rk[0]*max(m, n)

                return( self.SubAssembly(A11, A12, A21, A22), self.SubAssembly(B11, B12, B21, B22), self.SubAssembly(M11_rk, M12_rk, M21_rk, M22_rk), False)

            m1, m2 = A11.shape[0], A21.shape[0] 
            n1, n2 = B11.shape[1], B12.shape[1]
            m, n = m1+m2, n1+n2

            M_int = np.zeros((m, n))
            M_int[:m1,:n1] = np.dot( A11, B11 ) 
            M_int[:m1,n1:] = np.dot( A12, B12 ) 
            M_int[m1:,:n1] = np.dot( A21, B21 ) 
            M_int[m1:,n1:] = np.dot( A22, B22 ) 

            A_svd, B_svd, k_svd = self.SvdRk2( M_int ) # Svd transformation and rank modification

            rk_svd = k_svd[0]*( m + n )

            rk11 = M11_rk[0]*m1*n1
            rk12 = M12_rk[0]*m1*n2
            rk21 = M21_rk[0]*m2*n1
            rk22 = M22_rk[0]*m2*n2

            if rk_svd < rk11 + rk12 + rk21 + rk22 : # lower rank with svd
                return( A_svd, B_svd, k_svd, True)
            
            self.CompCoefSafeOpti += rk11 + rk12 + rk21 + rk22

            return( self.SubAssembly(A11, A12, A21, A22), self.SubAssembly(B11, B12, B21, B22), self.SubAssembly(M11_rk, M12_rk, M21_rk, M22_rk), False )

    

    def PlotBlockTree(self, M, posi, posj, pos1, pos2) :
        """
        PlotBlockTree(M, posi, posj, pos1, pos2) - function to contour the subblock in matrix imshow
        """

        if (type(M[0]) == float) or (type(M[0]) == int) :

            m, n = self.DimSLevel1[self.Nlevel][pos1], self.DimSLevel2[self.Nlevel][pos2]
            plt.plot([posi,posi+n,posi+n,posi,posi], [posj,posj,posj+m,posj+m,posj], 'k', linewidth=1)
            return(posi+n, posj+m)

        self.Nlevel += 1

        posi11, posj11 = self.PlotBlockTree(M[0][0], posi, posj, pos1*2, pos2*2)
        _, _ = self.PlotBlockTree(M[0][1], posi11, posj, pos1*2, pos2*2+1)
        posi21, _ = self.PlotBlockTree(M[1][0], posi, posj11, pos1*2+1, pos2*2)
        posi22, posj22 = self.PlotBlockTree(M[1][1], posi21, posj11, pos1*2+1, pos2*2+1)

        self.Nlevel -= 1

        return(posi22, posj22)
    

    def ProductVect(self, alpha, beta, M, x, b, pos1, pos2, root=False) :

        """ 
        ProductVect(alpha, beta, M, x, b, pos1, pos2) - product H_matrix (alpha)*M.x + (beta)*b = b
        --- input : 
        -      alpha, beta, coefficients for the matrix and the vector product
        -      M, H_matrix
        -      x, vector multiplied to the right
        -      b, output vector, needed for the incrementation
        -      pos1, pos2, relative positions in the cluster tree
        -      root (dafault value False)

        --- output :
        -      b, vector incremented 

        --- details :
        -      product defined by recursive call,
        -      and defined by subblock multiplication
        """
        
        if root == True :
            b = beta*b

        if (type(M[0][0]) == np.float64) or (type(M[0][0]) != np.float32) :
            return( alpha*np.dot(M,x) + b )
        
        # if (type(M[0][0]) != np.float64) or (type(M[0][0]) != np.float32):

        b_out = np.zeros( len(b) )

        self.Nlevel += 1 

        S1_2, S2_2 = self.DimSLevel1[self.Nlevel][pos1*2], self.DimSLevel2[self.Nlevel][pos2*2]
        x1, x2 = x[:S2_2], x[S2_2:]
        b1, b2 = b[:S1_2], b[S1_2:]

        b1 = self.ProductVect(alpha, 1, M[0][0], x1, b1, pos1*2, pos2*2)
        b1 = self.ProductVect(alpha, 1, M[0][1], x2, b1, pos1*2, pos2*2+1)
        b2 = self.ProductVect(alpha, 1, M[1][0], x1, b2, pos1*2+1, pos2*2)
        b2 = self.ProductVect(alpha, 1, M[1][1], x2, b2, pos1*2+1, pos2*2+1)
        
        self.Nlevel -= 1

        b_out[:S1_2] = b1
        b_out[S1_2:] = b2
        
        return( b_out )
        
    
    def ProductVect_AB(self, alpha, beta, A, B, x, b, pos1, pos2, root=False) : 

        """ 
        ProductVect_AB(alpha, beta, A, B, x, b, pos1, pos2) - product H_matrix (alpha)*(A.B).x + (beta)*b = b
        --- input : 
        -      same input as for ProductVect function 
        -      A, B, H_matrices 

        --- detail : same procedure as for the function ProductVect
        -      with the product (A.B) instead of using a single subblock H_matrix
        -      for admissible clusters
        """
        if root == True : 
            b = beta*b

        if self.Mfullblock_test(A) :

            if (len(B) == 0 ) : 
                return( alpha*np.dot(A, x) + b)
            return( alpha*np.dot( np.dot(A, B) , x ) + b ) 
        
        # if not self.Mfullblock_test(A) :

        b_out = np.zeros( len(b) )

        self.Nlevel += 1 

        S1_2, S2_2 = self.DimSLevel1[self.Nlevel][pos1*2], self.DimSLevel2[self.Nlevel][pos2*2]
        x1, x2 = x[:S2_2], x[S2_2:]
        b1, b2 = b[:S1_2], b[S1_2:]
        
        b1 = self.ProductVect_AB(alpha, 1, A[0][0], B[0][0], x1, b1, pos1*2, pos2*2)
        b1 = self.ProductVect_AB(alpha, 1, A[0][1], B[0][1], x2, b1, pos1*2, pos2*2+1)
        b2 = self.ProductVect_AB(alpha, 1, A[1][0], B[1][0], x1, b2, pos1*2+1, pos2*2)
        b2 = self.ProductVect_AB(alpha, 1, A[1][1], B[1][1], x2, b2, pos1*2+1, pos2*2+1)
        
        self.Nlevel -= 1

        b_out[:S1_2] = b1
        b_out[S1_2:] = b2
        
        return( b_out )
    


    def Mfullblock_test(self, M) :
        """
        Mfullblock_test - function to test if M is a subblock 
        --- detail : if true M is composed of float
        -       if not M have other hierarchie layer
        """
        if (type(M[0][0]) == np.float64) or (type(M[0][0]) == np.float32) or (type(M[0][0]) == float):
            return(True)
        return(False)


    def DimSLevel3_init(self, DimSLevel3) :
        """
        DimSLevel3_init(DmimSlevel3) - function to add another hierarchie structure 
        --- to use before hierarchie product
        """
        self.DimSLevel3 = DimSLevel3
    

    def SplitMat(self, M, pos1, pos2, i=1) :
        """
        SplitMat - function to split a subblock at its next hierarchie level
        --- useful for addition of H_matrices with different structure
        -          and for product , even if the structure are the same
        """
        if i == 1 :
            S1, S2 = self.DimSLevel1[self.Nlevel][2*pos1], self.DimSLevel2[self.Nlevel][2*pos2]
        elif i == 2 :
            S1, S2 = self.DimSLevel2[self.Nlevel][2*pos1], self.DimSLevel3[self.Nlevel][2*pos2]
        elif i == 3 :
            S1, S2 = self.DimSLevel1[self.Nlevel][2*pos1], self.DimSLevel3[self.Nlevel][2*pos2]
        
        M11 = M[:S1,:S2]
        M12 = M[:S1,S2:]
        M21 = M[S1:,:S2]
        M22 = M[S1:,S2:]
        return( np.array([[M11, M12], [M21, M22]])) #, dtype=np.ndarray) )


    
    def AddHmat(self, A, B, pos1, pos2, i=1) :
        """
        AddHmat( M1, M2, pos1, pos2) - H_addition of H_matrices (M1, M2)
        --- input : M1, M2 two H_matrices
        -      pos1, pos2, relative positions in the cluster trees

        --- output : H_matrix , with the finest structure possible 
        -      between M1, an M2 on theirs each subblock

        --- details : useful for the H_product (ProdHmat)
        """

        if not (self.Mfullblock_test(A)*self.Mfullblock_test(B)) :

            self.Nlevel += 1
            if (self.Mfullblock_test(A) == False) and (self.Mfullblock_test(B) == True):
                B_s = self.SplitMat(B, pos1, pos2, i)
                A_s = A.copy()
            if (self.Mfullblock_test(A) == True) and (self.Mfullblock_test(B) == False):
                B_s = B.copy()
                A_s = self.SplitMat(A, pos1, pos2, i)
            if (self.Mfullblock_test(A) == False) and (self.Mfullblock_test(B) == False) :
                A_s = A.copy()
                B_s = B.copy()
            

            C11 = self.AddHmat(A_s[0][0], B_s[0][0], 2*pos1, 2*pos2, i)
            C21 = self.AddHmat(A_s[1][0], B_s[1][0], 2*pos1+1, 2*pos2, i)
            C12 = self.AddHmat(A_s[0][1], B_s[0][1], 2*pos1, 2*pos2+1, i)
            C22 = self.AddHmat(A_s[1][1], B_s[1][1], 2*pos1+1, 2*pos2+1, i)

            self.Nlevel -= 1
            return( np.array([[C11, C12], [C21, C22]] ))#, dtype=np.ndarray))
        
        return(A + B)


    def ProductHmat(self, alpha, beta, M1, M2, C, pos1, pos2, pos3) :

        """
        ProductHmat(M1, M2, pos1, pos2, pos3) - product for H_matrices
        --- input : M1, M2, H_matrices
        -      pos1, pos2, pos3, relatives positions in the cluster trees

        --- output : C another H_matrix

        --- detail : 
        -      require tree different sets of element (explaining pos3)
        -      pos1, and self.DimSlevel1 for the line element set of M1
        -      pos2, and self.DimSlevel2 for the column element set of M1, and line element set of M2
        -      pos3, and self.DimSlevel3 for the column element set of M2

        -      recursive call, if the two subblock to multiply are hierarchicly decomposed 
        -      the product is decomposed with its subblocks
        -      [[A11, A12], [A21, A22]] . [[B11, B12], [B21, B22]] 
        -        = [[A11.B11 + A12.B21, A11.B21 + A12.B22], 
        -           [A21.B11 + A22.B12, A21.B21 + A22.B22]]
        """

        if all([self.Mfullblock_test(M1),self.Mfullblock_test(M2)]) :
            return( alpha*np.dot(M1, M2) + beta*C )
        
        elif any([self.Mfullblock_test(M1), self.Mfullblock_test(M2)]) :
            
            if not self.Mfullblock_test(M1) :
                M1asse = self.AssembleBlockTree(M1, pos1, pos2)
                return( alpha*np.dot(M1asse, M2) + beta*C )
            
            M2asse = self.AssembleBlockTree(M2, pos2, pos3, i=1)
            return( alpha*np.dot(M1, M2asse) + beta*C )

        self.Nlevel += 1

        C = self.SplitMat(C, pos1, pos3, i=3)

        C11_1 = self.ProductHmat(alpha, beta, M1[0][0], M2[0][0], C[0][0], 2*pos1, 2*pos2, 2*pos3) 
        C11_2 = self.ProductHmat(alpha, beta, M1[0][1], M2[1][0], C[0][0], 2*pos1, 2*pos2+1, 2*pos3) 
        C11 = self.AddHmat(C11_1, C11_2, 2*pos1, 2*pos3, 3)

        C21_1 = self.ProductHmat(alpha, beta, M1[1][0], M2[0][0], C[1][0], 2*pos1+1, 2*pos2, 2*pos3) 
        C21_2 = self.ProductHmat(alpha, beta, M1[1][1], M2[1][0], C[1][0], 2*pos1+1, 2*pos2+1, 2*pos3)
        C21 = self.AddHmat(C21_1, C21_2, 2*pos1+1, 2*pos3, 3)

        C12_1 = self.ProductHmat(alpha, beta, M1[0][0], M2[0][1], C[0][1], 2*pos1, 2*pos2, 2*pos3+1)
        C12_2 = self.ProductHmat(alpha, beta, M1[0][1], M2[1][1], C[0][1], 2*pos1, 2*pos2+1, 2*pos3+1)
        C12 = self.AddHmat(C12_1, C12_2, 2*pos1, 2*pos3+1, 3)

        # print('Nlevel : ' + str(self.Nlevel) + '\n')
        # print('C_11 : ' + str(C[1][1]) + '\n')
        # print('M1_21 : ' + str(M1[1][0]) + '\n')
        # print('M2_12 : ' + str(M2[0][1]) + '\n')
        C22_1 = self.ProductHmat(alpha, beta, M1[1][0], M2[0][1], C[1][1], 2*pos1+1, 2*pos2, 2*pos3+1)
        C22_2 = self.ProductHmat(alpha, beta, M1[1][1], M2[1][1], C[1][1], 2*pos1+1, 2*pos2+1, 2*pos3+1)
        C22 = self.AddHmat(C22_1, C22_2, 2*pos1+1, 2*pos3+1, 3)

        self.Nlevel -= 1
        return( np.array([[C11, C12], [C21, C22]] ))#, dtype=np.ndarray))

    
    # def LUpreHmat(self, M, pos1, pos2) :

    #     """
    #     LUpreHmat(M, pos1, pos2) - function for LU decomposition

    #     --- input : M H_matrix
    #     -      pos1, pos2, relative position of subblock in the cluster tree

    #     --- ouput : L, U LU decomposition in hierarchical format

    #     """

    #     if self.Mfullblock_test(M) :
    #         P, L, U = sclin.lu(M, permute_l=False)
    #         return(np.dot(P, L), U)
        
    #     self.Nlevel += 1

    #     L11, U11 = self.LUpreHmat(M[0][0], 2*pos1, 2*pos2)

    #     U12 = self.ForwardSolveLU(L11, M[0][1], 2*pos1, 2*pos2+1)
    #     L21 = self.BackwardSolveLU(U11, M[1][0], 2*pos1+1, 2*pos2)

    #     M[1][1] = self.ProductHmat(-1, 1, L21, U12, M[1][1], 2*pos1+1, 2*pos2, 2*pos2+1)
    #     L22, U22 = self.LUpreHmat(M[1][1], 2*pos1+1, 2*pos2+1)

    #     self.Nlevel -= 1

    #     return(np.array([[L11, []], [L21, L22]]), 
    #         np.array([[U11, U12, [[], U22]])) #, dtype=np.ndarray) )


    # def ForwardSolveLU(self, L, M, pos1, pos2) :
    #     return()

    # def BackwardSolveLU(self, L, M, pos1, pos2) :
    #     return()


    


    

