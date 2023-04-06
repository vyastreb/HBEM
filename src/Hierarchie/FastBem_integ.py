
import numpy as np 
import os 
import numpy.linalg as lin
import time 
import platform


def load_cluster_(cluster) :

    if type(cluster) == dict :
        return( cluster["STotElem"], cluster["STot"], cluster["SLevel"], cluster["SizeLevel"], 
            cluster["mesh"], cluster["mesh_size"], cluster["X0_mesh"], len( cluster["mesh"] ) )
    
    return(cluster.STotElem, cluster.STot, cluster.SLevel, cluster.SizeLevel,
        cluster.mesh, cluster.mesh_size, cluster.X0_mesh, cluster.Ne)


# from BEM_modul_v2.Core.tools import x_gap
def x_gap(X, Y) :
    return(lin.norm(X - Y))



class Bem_integ( object ) :


    def __init__(self, Cluster1, Cluster2, sing=False) :
        
        ## input
        self.sing = sing
        self.STotElem1, self.STot1, self.SLevel1, self.SizeLevel1, _, self.mesh_size1, self.X0_mesh1, self.Ne1 = load_cluster_(Cluster1)
        self.STotElem2, self.STot2, self.SLevel2, self.SizeLevel2, self.mesh2, self.mesh_size2, self.X0_mesh2, self.Ne2 = load_cluster_(Cluster2)

        ## V5 scipy, triangle, PART (gaussian both latter)
        self.error_S = 10e-10
        self.error_X = 10e-10
        self.error_the = 10e-10
        
        ## profiling the time for each computation
        self.t_ns = 0
        self.t_6pt = 0
        self.t_gauss = 0
        self.t_tri = 0
        self.t_sing = 0

        ## number of integrals computed
        self.N_6pt_int = 0        
        self.N_tri_int = 0
        self.N_gauss_int = 0
        self.N_sing_int = 0
               

        ### initegration element O(Ne)
        # gauss point and weight
        self.Ng_max_quad = 6 ## for singular integral 
        self.Ng_IS = [7, 16, 19] # [6, 7, 16, 19, 28, 16, 19, 28]
        self.read_gauss_file() # Coord_gauss_tri.npz in Model4

        # setup for integration on elem far away Npt_gauss=4
        # t_ = time.time()
        # self.mesh_size2 = self.init_size_elem( self.mesh2 )
        self.Dmax2 = np.max( self.mesh_size2 )
        self.dmin2 = 2.37*self.Dmax2

        self.G_matrix_lin = np.array([[-1, 1, 0], [-1, 0, 1]])
        self.G_mesh2 = self.init_G_mesh( self.mesh2 )
        self.G_mesh2_6pt = self.init_G_mesh_6pt( self.mesh2 )

        self.init_N_6pt()
        self.X_mesh2_6pt = self.init_X_mesh_6pt( self.mesh2 )

        # setup for integration on singular 
        if self.sing == True :
            self.init_param_triangle() 

        # vectorisation of function
        self.F_integ_sing_vect = np.vectorize(self.F_integ_sing)
        self.F_integ_nsing_vect = np.vectorize(self.F_integ_nsing)




    ## read file gauss coords and weights
    def read_gauss_file(self) :
		
        os.chdir( '../Integrate_v0' )
        file_r = np.load('Gauss_coord_tri.npz',allow_pickle=True)
        Coord_load, Weight_load = file_r['Coords'], file_r['Weights']
        file_r.close()

        self.n_6pt = np.array( Coord_load[0] )
        self.gauss_weight_6pt = np.array( Weight_load[0] )

        self.gauss_coords_flat = np.zeros((len(Coord_load)-1, len(Coord_load[-1]), 2))
        self.gauss_weights_flat = np.zeros((len(Coord_load)-1, len(Coord_load[-1])))
        for k in range(len(Coord_load)-1) :
            for i in range( self.Ng_IS[k] ):
                self.gauss_coords_flat[k][i] = np.array(Coord_load[k+1][i])
                self.gauss_weights_flat[k][i] = Weight_load[k+1][i]

        ## if self.sing
        os.chdir(self.path_file_gauss)
        file_r = np.load('Gauss_coord_quad.npz',allow_pickle=True)
        Coord_load, Weight_load = file_r['Coords'], file_r['Weights']
        file_r.close()

        self.gauss_coords_quad = np.zeros((3, self.Ng_max_quad))
        self.gauss_weights_quad = np.zeros((3, self.Ng_max_quad))
        for k in range(3, 6) :
            for i in range(k + 1) :
                self.gauss_coords_quad[k-3][i] = Coord_load[k][i]
                self.gauss_weights_quad[k-3][i] = Weight_load[k][i]
        
        #if self.sing == True :
        self.gauss_coords_quad_s = np.array( Coord_load[self.Ng_max_quad-1] )
        self.gauss_weights_quad_s = np.array( Weight_load[self.Ng_max_quad-1] )

        del Coord_load, Weight_load


    ## jacobian value for each element and at each integration point
    def init_G_mesh(self, mesh) :
        return( np.array( [ np.abs(lin.det( np.dot( self.G_matrix_lin, mesh[k] ) ) ) for k in range(len(mesh))] ) )

    def init_G_mesh_6pt(self, mesh) :
        G_mesh_6pt = np.zeros((len(mesh),6))
        for k in range(len(mesh)) :
            G_mesh_6pt[k] = np.array([ self.gauss_weight_6pt[i]*np.abs(lin.det( np.dot( self.G_matrix_lin, mesh[k] ))) for i in range(6) ] )
        return( G_mesh_6pt )

    ## shape function for an element with 6pt integration
    def init_N_6pt(self) :
        self.N_6pt = np.array([self.N_tot_def(n_6pt_i) for n_6pt_i in self.n_6pt])

    ## coordinate for each element at 6pt intergration
    def init_X_mesh_6pt(self, mesh) :
        return( np.array([ [ np.dot( self.N_6pt[i], mesh[k] ) for i in range(6)] for k in range(len(mesh))]) )


    ## function definition paramters for singular integrals and triangles separation
    def init_param_triangle(self) :
        
        self.h_list = np.array([1/3, np.sqrt(2)/6, 1/3])
        self.alpha1_list = np.array([-3*np.pi/4, -(np.pi)/2 + np.arctan(2), (np.pi)-np.arctan(2)])

        a2 = (np.pi)/4 + np.arctan(1/2)
        a3 = (np.pi)/2 - np.arctan(1/2)
        self.alpham_list = np.array([np.pi/4, a2, a3])
        self.dthe_list = np.array([(np.pi)/4 + np.arctan(2), 2*a2, (np.pi)/4 + np.arctan(2)])

        self.n_corner = np.array([[0,0], [1,0], [0,1]])

        self.L = np.array([ [[np.cos(self.alpha1_list[k]), -np.sin(self.alpha1_list[k])],
            [np.sin(self.alpha1_list[k]),np.cos(self.alpha1_list[k])]] for k in range(3)] )

        self.Npt_sing = self.Ng_max_quad*self.Ng_max_quad*3
        self.gauss_coords_sing = np.zeros((self.Npt_sing,2))
        self.gauss_weights_sing = np.zeros((self.Npt_sing))

        ind = 0
        for k in range(3) :

            L_k = self.L[k]

            for ind1 in range(self.Ng_max_quad) :

                the = self.dthe_list[k]*(1 + self.gauss_coords_quad_s[ind1])/2
                dthe = self.dthe_list[k]*self.gauss_weights_quad_s[ind1]/2
                rho_max_i = self.h_list[k]/(np.cos(the-self.alpham_list[k]))

                for ind2 in range(self.Ng_max_quad) :
                    
                    rho = rho_max_i*(1 + self.gauss_coords_quad_s[ind2])/2
                    drho = rho_max_i*self.gauss_weights_quad_s[ind2]/2
                    
                    e = np.array([rho*np.cos(the), rho*np.sin(the)])
                    n_a = np.dot(L_k,e) + np.array([1/3, 1/3])
                    
                    self.gauss_coords_sing[ind] = n_a
                    self.gauss_weights_sing[ind] = rho*drho*dthe

                    ind += 1


    ## gap function for green function
    def x_gap(self, X1, X2) :
        return( lin.norm( X1 - X2 ) )
    
    def x_gap_array(self, X1, X2) :
        return( lin.norm( X1 - X2 , axis=1) )
                

    ## shape functions for quadrangular element quadratic 
    def N_tot_def(self, n) :
        n1, n2 = n[0], n[1]
        return( np.array([ (1-n1-n2), n1, n2 ] ) )
    
    def dN_tot_def(self) :
        return( np.array( [[-1, 1, 0], [-1, 0, 1]]))
    

    ## interpolation function and jacobian
    def X_inter(self, n, elem) :
        return( np.dot( self.N_tot_def(n).T, elem) )
    
    def J_inter(self, n, elem) :
        return( np.dot( self.dN_tot_def, elem).T )

    def J_inter_norm(self, n, elem) :
        return( np.abs(lin.det(self.J_inter(n, elem))) )
    

    ## kernel function - green function
    def G(self, X1, X2) :
        return( 1/(4*np.pi*x_gap(X1,X2)) )
    
    def G_array(self, X1, X2) :
        return( 1/(4*np.pi*self.x_gap_array(X1,X2)) )

    ## returning the number of useful Gauss point needed
    def IS_def(self, d, D) :
        IS = int( 2.37*D/d ) - 1
        return( min( max(1,IS), 8 ) )


    ######## PRATT method
    ## define xs coordinates 
    def xs_segment_def(self, X1, X2, X_) :
        Xv = (X2 - X1) / x_gap(X1, X2)
        Bh = np.dot( (X_ - X1).T, Xv )
        return( X1 + Bh * Xv )
    

    # controling if the point is in the segment for the both coordinate
    def xs_test_in_segment( self, X1, X2, Xs_init) :
        if X1[0] != X2[0] :
            return( np.abs( Xs_init[1] - (X1[1] + (X2[1]-X1[1])*(X1[0]-Xs_init[0])/(X1[0]-X2[0])) ) <= self.error_S ) 
        return( np.abs( Xs_init[0] - (X1[0] + (X2[0]-X1[0])*(X1[1]-Xs_init[1])/(X1[1]-X2[1])) ) <= self.error_S ) 
        

    # if the Xs point found is out, which corner point is the closest
    def xs_out_segment(self, X1, X2, Xs_init) :

        if X2[1] != X1[1] :
            if  Xs_init[1] >= np.max([X2[1], X1[1]]) and np.max([X2[1], X1[1]]) == X2[1]:
                return( X2 )
            if  Xs_init[1] >= np.max([X2[1], X1[1]]) and np.max([X2[1], X1[1]]) == X1[1]:
                return( X1 )
            if  Xs_init[1] <= np.min([X2[1], X1[1]]) and np.min([X2[1], X1[1]]) == X2[1]:
                return( X2 )
            if  Xs_init[1] <= np.min([X2[1], X1[1]]) and np.min([X2[1], X1[1]]) == X1[1]:
                return( X1 )
        else :
            if  Xs_init[0] >= np.max([X2[0], X1[0]]) and np.max([X2[0], X1[0]]) == X2[0]:
                return( X2 )
            if  Xs_init[0] >= np.max([X2[0], X1[0]]) and np.max([X2[0], X1[0]]) == X1[0]:
                return( X1 )
            if  Xs_init[0] <= np.min([X2[0], X1[0]]) and np.min([X2[0], X1[0]]) == X2[0]:
                return( X2 )
            if  Xs_init[0] <= np.min([X2[0], X1[0]]) and np.min([X2[0], X1[0]]) == X1[0]:
                return( X1 )
        

    ## function to find the xs coordinate 
    def xs_def(self, elem, X_int) :

        X_corner = [elem[0], elem[1], elem[2]]
        #print( X_corner )
        Xs_segment = []
        for i in range(3):
            Xi, Xi1 = X_corner[i], X_corner[(i+1)%3]
            Xsi_init = self.xs_segment_def(Xi, Xi1, X_int)
            
            if Xsi_init[0] >= min(Xi1[0], Xi[0]) and Xsi_init[0] <= np.max([Xi1[0], Xi[0]]) :
                if Xsi_init[1] >= np.min([Xi1[1], Xi[1]]) and Xsi_init[1] <= np.max([Xi1[1], Xi[1]]) :
                    Xs_segment.append( Xsi_init )
                elif np.abs(Xi1[1] - Xi[1]) <= self.error_X and np.abs(Xsi_init[1] - Xi[1]) <= self.error_X :
                    Xs_segment.append( Xsi_init )
                else :
                    Xs_segment.append( self.xs_out_segment(Xi, Xi1, Xsi_init) )
            elif np.abs(Xi1[0] - Xi[0]) <= self.error_X and np.abs(Xsi_init[0] - Xi[0]) <= self.error_X :
                Xs_segment.append( Xsi_init )
            else :
                Xs_segment.append( self.xs_out_segment(Xi, Xi1, Xsi_init) )
        
        Xs = Xs_segment[0]
        
        d = x_gap(np.array(Xs), np.array(X_int) ) # initialize      
        for i in range(1,3) :
            Xsi = Xs_segment[i]
            di = x_gap(np.array(Xsi), np.array(X_int) )
            if di < d : 
                d = di 
                Xs = Xsi
        del Xsi , di
        
        ind_segment_list = []
        for i in range(3) :
            Xi , Xi1 = X_corner[i], X_corner[(i+1)%3]
            if self.xs_test_in_segment(Xi1, Xi, Xs) :
                ind_segment_list.append( i+1 )
        
        return( Xs, d, ind_segment_list)
    

    ## function to find ns local coordinate for Xs in elem  
    def ns_def_lin(self, elem, Xs, ind_segment) :
        
        if ind_segment[0] == 1 :
            if elem[0][0] != elem[1][0] :
                return(  [(elem[0][0]-Xs[0])/(elem[0][0]-elem[1][0]),0] )
            else : 
                return(  [(elem[0][1]-Xs[1])/(elem[0][1]-elem[1][1]),0] )
            
        elif ind_segment[0] == 2 :
            if elem[1][1] != elem[2][1] :
                return(  [(elem[2][1]-Xs[1])/(elem[2][1]-elem[1][1]), 1-(elem[2][1]-Xs[1])/(elem[2][1]-elem[1][1])] )
            else :
                return(  [(elem[2][0]-Xs[0])/(elem[2][0]-elem[1][0]), 1-(elem[2][0]-Xs[0])/(elem[2][0]-elem[1][0])] )                
        
        elif ind_segment[0] == 3 :
            if elem[2][0] != elem[0][0] :
                return(  [0, (elem[0][0]-Xs[0])/(elem[0][0]-elem[2][0])] )
            else : 
                return(  [0, (elem[0][1]-Xs[1])/(elem[0][1]-elem[2][1])] )
                

    def Delta_j(self, n_b, ind_segment) :
        n1, n2 = n_b[0], n_b[1]

        sign = [1, 1, 1]
        for ind in ind_segment :
            sign[ind-1] = 0
        #print( sign , ind_segment )

        if len(ind_segment) == 1 :
            if sign[0] == 0 :

                dthe2 = np.arctan(1/n1)
                #print( dthe2 , (np.pi)/2 )
                dthe1 = (np.pi) - dthe2

                h1 = (1-n1)*np.sqrt(2)/2
                
                h_list = [h1, n1]
                d_the_list = [dthe1, dthe2]

                alpha_list = [(np.pi)/4, dthe2]

                L1 = [[1, 0],[0, 1]]
                L2 = [[np.cos(dthe1), -np.sin(dthe1)],[np.sin(dthe1), np.cos(dthe1)]]
                L_list = [L1, L2]

                D_list = [1, 1]
                return( L_list, D_list, d_the_list, alpha_list, h_list)
            
            elif sign[1] == 0 :
                h_list = [n2, n1]

                a1 = np.arctan(n1/n2)
                #a2 = np.arctan((1-n1)/n2)
                alpha_list = [a1, (np.pi)/4]

                al1 = -((np.pi)/2 + a1) 
                al2 = 3*(np.pi)/4
                L1 = [[np.cos(al1), -np.sin(al1)],[np.sin(al1), np.cos(al1)]]
                L2 = [[np.cos(al2), -np.sin(al2)],[np.sin(al2), np.cos(al2)]]
                L_list = [L1, L2]

                d_the_list = [a1 + (np.pi)/4, 3*(np.pi)/4 - a1]
                D_list = [1, 1]
                return( L_list, D_list, d_the_list, alpha_list, h_list)
            
            elif sign[2] == 0 :
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

                D_list = [(1-n2)/2, n2/2]
                D_list = [1, 1]
                return( L_list, D_list, d_the_list, alpha_list, h_list)
        else :
            if all(ind_segment) in [1, 2]  :
                h_list = [1]
                d_the_list = [(np.pi)/4]
                alpha_list = [3*(np.pi)/4]
                L_list = [[[-np.sqrt(2)/2,-np.sqrt(2)/2],[np.sqrt(2)/2, -np.sqrt(2)/2]]]
                return( L_list, [1], d_the_list, alpha_list, h_list)
            if all(ind_segment) in [2, 3] :
                h_list = [1]
                d_the_list = [np.pi/4]
                alpha_list = [-(np.pi)/2]
                L_list = [[[0,1], [-1,0]]]
                return( L_list, [1], d_the_list, alpha_list, h_list)
            if all(ind_segment) in [1, 3] :
                h_list = [np.sqrt(2)/2]
                d_the_list = [(np.pi)/2]
                alpha_list = [(np.pi)/4]
                L_list = [[[1,0], [0,1]]]
                return( L_list, [1], d_the_list, alpha_list, h_list)
    

    ## final function for integration - Gaussian quadrature 
    def coef_f_gauss_6pt(self, j, X_int) :
        t_ = time.time()
        f = np.dot( self.G_array(self.X_mesh2_6pt[j], X_int),(self.G_mesh2_6pt[j]).T ) 
        self.t_6pt += time.time() - t_ 
        return( f )

    
    def coef_f_gauss(self, elem, ind_elem, X_int, Ng) :

        t_ = time.time()
        len_gauss_coords = self.Ng_IS[Ng-2]
        X_mesh_Ng = np.array([ self.X_inter( self.gauss_coords_flat[Ng-2][k], elem ) for k in range(len_gauss_coords) ])
        G_w_mesh_Ng = np.array([ self.gauss_weights_flat[Ng-2][k]*self.G_mesh2[ind_elem] for k in range(len_gauss_coords) ])
        
        f = np.dot( self.G_array(X_mesh_Ng, X_int),G_w_mesh_Ng.T ) 
        self.t_gauss += time.time() - t_
        return( f )

    
    def coef_f_triangle_gauss(self, elem, ind_elem, X_int, Ng) :

        t_ = time.time()

        Xs, d, ind_segment = self.xs_def(elem, X_int)
        #print( Xs, d, ind_segment)
        ns = self.ns_def_lin(elem, Xs, ind_segment)
        L, D, d_the, alpha, h = self.Delta_j(ns, ind_segment)
        del d

        I = 0
        G = self.G_mesh2[ind_elem]

        for j in range(len(L)) :
            L_j, d_the_j, alpha_j, h_j = L[j], d_the[j], alpha[j], h[j]

            #print( alpha_j, h_j , d_the_j, L_j)
            I_j = 0
            for k in range(Ng+1) :
                gk = self.gauss_coords_quad[Ng-3][k]
                wk = self.gauss_weights_quad[Ng-3][k]
                the_jk = (d_the_j/2)*(1+gk)

                rho_max_jk = h_j/(np.cos(the_jk-alpha_j))
                I_k = 0
                for l in range(Ng+1) :
                    gl = self.gauss_coords_quad[Ng-3][l]
                    wl = self.gauss_weights_quad[Ng-3][l]
                    rho_jkl = (rho_max_jk/2)*(1+gl) 
                    
                    e_a = rho_jkl*np.array([np.cos(the_jk), np.sin(the_jk)])

                    n_a = np.dot(np.array(L_j), e_a) + np.array(ns)

                    #G_jkl = self.J_inter_norm(n_a, elem)
                    X_jkl = self.X_inter(n_a, elem)
                    
                    #plt.plot( X_jkl[0], X_jkl[1], 'x', color='violet')
                    f_jkl = self.G(X_jkl, X_int)

                    I_k += (wl/2)*d_the_j*f_jkl*G*rho_jkl

                I_j += (wk/2)*I_k*rho_max_jk
            I += I_j
        
        self.t_tri += time.time() - t_
        return( I )


    def coef_f_singular_gauss(self, elem, ind_elem, X_int) :

        t_ = time.time()
        X_mesh_sing = np.array([ self.X_inter( self.gauss_coords_sing[k], elem ) for k in range(self.Npt_sing) ])
        G_w_mesh_sing = np.array([ self.gauss_weights_sing[k]*self.G_mesh2[ind_elem] for k in range(self.Npt_sing) ])
        
        f = np.dot( self.G_array(X_mesh_sing, X_int),G_w_mesh_sing.T )
        self.t_sing += time.time() - t_
        return( f )
    

    def F_integ_sing(self, i, j) : 
        
        if i == j :
            self.N_sing_int += 1
            return( self.coef_f_singular_gauss(self.mesh2[j], j, self.X0_mesh2[i]) )

        d_ij = x_gap(self.X0_mesh2[i], self.X0_mesh2[j])
        
        if d_ij >= self.dmin2 :
            self.N_6pt_int += 1
            return( self.coef_f_gauss_6pt(j, self.X0_mesh2[i]) )

        IS = self.IS_def(d_ij, self.mesh_size2[j] )
        if IS == 1 :
            self.N_6pt_int += 1
            return( self.coef_f_gauss_6pt(j, self.X0_mesh2[i]) )
        elif IS < 5 :
           self.N_gauss_int += 1
           return( self.coef_f_gauss(self.mesh2[j], j, self.X0_mesh2[i], IS ) )
        # self.N_gauss_int += 1
        # return( self.coef_f_gauss(self.mesh2[j], j, self.X0_mesh2[i], min(IS, 4) ) )
        self.N_tri_int += 1
        return( self.coef_f_triangle_gauss(self.mesh2[j], j, self.X0_mesh2[i], max(3, IS-3) ) )



    def F_integ_nsing(self, i, j) : 

        d_ij = x_gap(self.X0_mesh1[i], self.X0_mesh2[j])

        if d_ij >= self.dmin2 :
            self.N_6pt_int += 1
            return( self.coef_f_gauss_6pt(j, self.X0_mesh1[i]) )

        IS = self.IS_def(d_ij, self.mesh_size2[j] )
        if IS == 1 :
            self.N_6pt_int += 1
            return( self.coef_f_gauss_6pt(j, self.X0_mesh1[i]) )
        elif IS < 5 :
            self.N_gauss_int += 1
            return( self.coef_f_gauss(self.mesh2[j], j, self.X0_mesh1[i], IS ) )

        self.N_tri_int += 1
        return( self.coef_f_triangle_gauss(self.mesh2[j], j, self.X0_mesh1[i], max(3, IS-3) ) )

        # self.N_gauss_int += 1
        # return( self.coef_f_gauss(self.mesh2[j], j, self.X0_mesh1[i], min(IS, 5) ) )

