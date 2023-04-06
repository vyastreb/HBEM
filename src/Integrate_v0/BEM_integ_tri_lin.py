
import numpy as np 
import os 
import numpy.linalg as lin
import time 


class BEM_integ_tri_lin :


    def __init__(self, mesh, path_save, filename_save) :

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
        
        self.Ne = len(mesh) #reading the mesh 
        print(" -- Ne : {} number of element".format(self.Ne))
        self.init_mesh(mesh) 
        self.init_X_integ()

        ### initegration element O(Ne)*
        # gauss point and weight
        self.Ng_max_quad = 6 ## for singular integral 
        self.Ng_IS = [7, 16, 19] # [6, 7, 16, 19, 28, 16, 19, 28]
        self.read_gauss_file() # Coord_gauss_tri.npz in Model4

        # setup for integration on elem far away Npt_gauss=4
        t_ = time.time()
        self.init_size_elem()

        self.G_matrix_lin = np.array([[-1, 1, 0], [-1, 0, 1]])
        self.init_G_mesh()
        self.init_G_mesh_6pt()

        self.init_N_6pt()
        self.init_X_mesh_6pt()
        self.t_build_6pt = time.time() - t_
        print( ' ---- construction of 4pt tools - time : {}'.format(self.t_build_6pt) )

        # setup for integration on singular 
        t_ = time.time()
        self.init_param_triangle() 
        self.t_build_sing = time.time() - t_
        print( ' ---- construction of sing tools - time : {}'.format(self.t_build_sing) )

        ## save input for saving 
        self.path_save = path_save
        self.filename_save = filename_save
        self.filename_save_npz = filename_save + '.npz'

        #print( ' --- D_287 {}, D_231 {}, d {}, IS {}'.format( self.size_elem[287], self.size_elem[231], self.x_gap(self.X_integ[287], self.X_integ[231]), self.N_gauss(self.x_gap(self.X_integ[287], self.X_integ[231]), self.size_elem[231])))

        
        print("\n###### Start building F matrix ########\n")
        self.build_F() #F matrix
        print('\n -- computation done times :\n ---- def t_6pt : {0}, N_6pt : {1} \n ---- integ t_gauss : {2}, N_gauss : {3} \n ---- integ t_tri : {4}, N_tri : {5} \n ---- integ t_sing : {6}, N_sing : {7}'.format(self.t_6pt, self.N_6pt_int, self.t_gauss, self.N_gauss_int, self.t_tri, self.N_tri_int, self.t_sing, self.N_sing_int))
        

    ## read file gauss coords and weights
    def read_gauss_file(self) :

        ## for the lower IS - integration on triangle element
        os.chdir('.')
        file_r = np.load('Gauss_coord_tri.npz',allow_pickle=True)
        Coord_load, Weight_load = file_r['Coords'], file_r['Weights']
        del file_r

        self.n_6pt = np.array( Coord_load[0] )
        #self.n_6pt = np.array([ [1/6,1/6], [1/6,4/6], [])
        self.gauss_weight_6pt = np.array( Weight_load[0] )

        self.gauss_coords_flat = np.zeros((len(Coord_load)-1, len(Coord_load[-1]), 2))
        self.gauss_weights_flat = np.zeros((len(Coord_load)-1, len(Coord_load[-1])))
        for k in range(len(Coord_load)-1) :
            for i in range( self.Ng_IS[k] ):
                self.gauss_coords_flat[k][i] = np.array(Coord_load[k+1][i])
                self.gauss_weights_flat[k][i] = Weight_load[k+1][i]
        print( self.gauss_coords_flat)
        print( self.gauss_weights_flat)

        del Coord_load, Weight_load


        ## for singuar integral - and the nearly singular 
        os.chdir('.')
        file_r = np.load('Gauss_coord_quad.npz',allow_pickle=True)
        Coord_load, Weight_load = file_r['Coords'], file_r['Weights']
        del file_r

        self.gauss_coords_quad = np.zeros((3, self.Ng_max_quad ))
        self.gauss_weights_quad = np.zeros((3, self.Ng_max_quad ))
        for k in range(3,6) :
            for i in range( k+1 ) :
                self.gauss_coords_quad[k-3][i] = Coord_load[k][i]
                self.gauss_weights_quad[k-3][i] = Weight_load[k][i]
        self.gauss_coords_quad_s = np.array( Coord_load[self.Ng_max_quad-1] )
        self.gauss_weights_quad_s = np.array( Weight_load[self.Ng_max_quad-1] )

        del Coord_load, Weight_load



    
    ## construction of vector coordinate 
    def init_mesh(self, mesh) :
        #self.mesh = np.zeros((self.Ne, 9,2))
        self.mesh = np.array([ [[mesh[k][i][1], mesh[k][i][2]] for i in range(1,4)] for k in range(self.Ne)] )
        print( ' --- construction self.mesh shape : {}'.format(self.mesh.shape) )


    
    def init_mesh_sym(self, mesh_sym) :
        #self.mesh = np.zeros((self.Ne, 9,2))
        self.mesh_sym = np.array([ [[mesh_sym[k][i][1], mesh_sym[k][i][2]] for i in range(1,4)] for k in range(self.Ne)] )
        print( ' --- construction self.mesh_new shape : {}'.format(self.mesh_sym.shape) )
    

    def init_X_integ(self) :
        self.X_integ = np.array([ np.dot( np.array([1/3, 1/3, 1/3]), self.mesh[i] ) for i in range(self.Ne)])
        #self.X_integ_4pt = np.array([ [self.mesh[i][4]] for i in range(self.Ne)])
        print( ' --- construction self.X_integ shape : {}'.format(self.X_integ.shape) )
        print( ' --- {}'.format(self.X_integ[0]))


    ## initialisation of the vector size
    def D_size_elem(self, elem) :
        #print( elem )
        return( max(self.x_gap(np.array(elem[0]), np.array(elem[1])), self.x_gap(np.array(elem[0]), np.array(elem[2])) ) )


    def init_size_elem(self) :
        self.size_elem = np.array([ self.D_size_elem(self.mesh[i]) for i in range(self.Ne)] )
        self.Dmax = np.max(self.size_elem)
        self.dmin = 2.37*self.Dmax


    ## jacobian value for each element and at each integration point
    def init_G_mesh(self) :
        self.G_mesh = np.array( [ np.abs(lin.det( np.dot( self.G_matrix_lin, self.mesh[k] ) ) ) for k in range(self.Ne)] )

    def init_G_mesh_6pt(self) :
        self.G_mesh_6pt = np.zeros((self.Ne,6))
        for k in range(self.Ne) :
            self.G_mesh_6pt[k] = np.array([ self.gauss_weight_6pt[i]*np.abs(lin.det( np.dot( self.G_matrix_lin, self.mesh[k] ))) for i in range(6) ] )


    ## shape function for an element with 4pt integration
    def init_N_6pt(self) :
        self.N_6pt = np.array([self.N_tot_def(n_6pt_i) for n_6pt_i in self.n_6pt])
        print( ' --- construction self.N_6pt shape : {}'.format( self.N_6pt.shape) )


    ## coordinate for each element at 4pt intergration
    def init_X_mesh_6pt(self) :
        self.X_mesh_6pt = np.array([ [ np.dot( self.N_6pt[i], self.mesh[k] ) for i in range(6)] for k in range(self.Ne)])
        print( ' --- construction self.mesh_6pt shape : {}'.format(self.X_mesh_6pt.shape) )


    ## function definition paramters for singular integrals and triangles separation
    def init_param_triangle(self) :
        
        self.h_list = np.array([1/3, np.sqrt(2)/6, 1/3])
        self.alpha1_list = np.array([-3*np.pi/4, -(np.pi)/2 + np.arctan(2), (np.pi)-np.arctan(2)])

        a2 = (np.pi)/4 + np.arctan(1/2)
        a3 = (np.pi)/2 - np.arctan(1/2)
        self.alpham_list = np.array([np.pi/4, a2, a3])
        self.dthe_list = np.array([(np.pi)/4 + np.arctan(2), 2*a2, (np.pi)/4 + np.arctan(2)])

        self.n_corner = np.array([[0,0], [1,0], [0,1]])

        self.L = np.array([ [[np.cos(self.alpha1_list[k]), -np.sin(self.alpha1_list[k])],[np.sin(self.alpha1_list[k]),np.cos(self.alpha1_list[k])]] for k in range(3)] )
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
        print( ' --- construction of param triangle self.gauss_coords_sing and self.gauss_weights_sing ')
    

    ## gap function for green function
    def x_gap(self, X1, X2) :
        #return( np.sqrt((X1[0]-X2[0])*(X1[0]-X2[0]) + (X1[1]-X2[1])*(X1[1]-X2[1])) )
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
    
    def G_inter(self, n, elem) :
        return( np.dot( self.dN_tot_def, elem).T )

    def G_inter_norm(self, n, elem) :
        return( np.abs(lin.det(self.G_inter(n, elem))) )
    

    ## kernel function - green function
    def H(self, X1, X2) :
        return( 1/(4*np.pi*self.x_gap(X1,X2)) )
    
    def H_array(self, X1, X2) :
        return( 1/(4*np.pi*self.x_gap_array(X1,X2)) )

    ## returning the number of useful Gauss point needed
    def IS_def(self, d, D) :
        IS = int( 2.37*D/d ) - 1
        return( min( max(1,IS), 8 ) )


    ######## PRATT method
    ## define xs coordinates 
    def xs_segment_def(self, X1, X2, X_) :

        xm, ym = X2[0], X2[1]
        xb, yb = X1[0], X1[1]
        xa, ya = X_[0], X_[1]

        xv = (xm-xb)/np.sqrt((xm-xb)*(xm-xb)+(ym-yb)*(ym-yb))
        yv = (ym-yb)/np.sqrt((xm-xb)*(xm-xb)+(ym-yb)*(ym-yb))

        BH = (xa-xb)*xv + (ya-yb)*yv
        xh = xb + BH*xv
        yh = yb + BH*yv

        del xm, ym, xa, ya, xb, yb
        del xv, yv, BH

        return( [xh, yh] )
    

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

        #Xe, Ye = self.read_coord_elem(elem)
        #print( elem )
        X_corner = [elem[0], elem[1], elem[2]]
        #print( X_corner )
        Xs_segment = []
        for i in range(3):
            Xi , Xi1 = X_corner[i], X_corner[(i+1)%3]
            Xsi_init = self.xs_segment_def(Xi, Xi1, X_int)
            
            if Xsi_init[0] >= np.min([Xi1[0], Xi[0]]) and Xsi_init[0] <= np.max([Xi1[0], Xi[0]]) :
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
        
        d = self.x_gap(np.array(Xs), np.array(X_int) ) # initialize      
        for i in range(1,3) :
            Xsi = Xs_segment[i]
            di = self.x_gap(np.array(Xsi), np.array(X_int) )
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
        return( np.dot( self.H_array(self.X_mesh_6pt[j], X_int),(self.G_mesh_6pt[j]).T ) )

    
    def coef_f_gauss(self, elem, ind_elem, X_int, Ng) :

        t_ = time.time()
        len_gauss_coords = self.Ng_IS[Ng-2]
        X_mesh_Ng = np.array([ self.X_inter( self.gauss_coords_flat[Ng-2][k], elem ) for k in range(len_gauss_coords) ])
        G_w_mesh_Ng = np.array([ self.gauss_weights_flat[Ng-2][k]*self.G_mesh[ind_elem] for k in range(len_gauss_coords) ])
        
        f = np.dot( self.H_array(X_mesh_Ng, X_int),G_w_mesh_Ng.T ) 
        self.t_gauss += time.time() - t_
        return( f )

        #return( np.dot( self.H_array(X_mesh_Ng, X_int),G_w_mesh_Ng.T )  )

    
    def coef_f_triangle_gauss(self, elem, ind_elem, X_int, Ng) :

        t_ = time.time()

        Xs, d, ind_segment = self.xs_def(elem, X_int)
        #print( Xs, d, ind_segment)
        ns = self.ns_def_lin(elem, Xs, ind_segment)
        L, D, d_the, alpha, h = self.Delta_j(ns, ind_segment)
        del d

        I = 0
        G = self.G_mesh[ind_elem]

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

                    #G_jkl = self.G_inter_norm(n_a, elem)
                    X_jkl = self.X_inter(n_a, elem)
                    
                    #plt.plot( X_jkl[0], X_jkl[1], 'x', color='violet')
                    f_jkl = self.H(X_jkl, X_int)

                    I_k += (wl/2)*d_the_j*f_jkl*G*rho_jkl

                I_j += (wk/2)*I_k*rho_max_jk
            I += I_j
        
        self.t_tri += time.time() - t_
        return( I )


    def coef_f_singular_gauss(self, elem, ind_elem, X_int) :

        t_ = time.time()
        X_mesh_sing = np.array([ self.X_inter( self.gauss_coords_sing[k], elem ) for k in range(self.Npt_sing) ])
        G_w_mesh_sing = np.array([ self.gauss_weights_sing[k]*self.G_mesh[ind_elem] for k in range(self.Npt_sing) ])
        
        f = np.dot( self.H_array(X_mesh_sing, X_int),G_w_mesh_sing.T )
        self.t_sing += time.time() - t_
        return( f )
        #return( np.dot( self.H_array(X_mesh_sing, X_int),G_w_mesh_sing.T ) )

        
    
    def F_integ(self, i, j, X_integ_i) : #, X_integ_4pt_i):
        if i == j :
            self.N_sing_int += 1
            return( self.coef_f_singular_gauss(self.mesh[j], j, X_integ_i) )

        else : # no singularity
            d_ij = self.x_gap(X_integ_i, self.X_integ[j])
            if d_ij >= self.dmin :
                self.N_6pt_int += 1
                return( self.coef_f_gauss_6pt(j, X_integ_i) )

            else :
                IS = self.IS_def(d_ij, self.size_elem[j] )
                if IS == 1 :
                    self.N_6pt_int += 1
                    return( self.coef_f_gauss_6pt(j, X_integ_i) )

                elif IS < 5 :
                    self.N_gauss_int += 1
                    return( self.coef_f_gauss(self.mesh[j], j, X_integ_i, IS ) )
                
                if IS == 5 :
                    self.N_tri_int += 1
                    return( self.coef_f_triangle_gauss(self.mesh[j], j, X_integ_i, 3 ) )
                else : 
                    self.N_tri_int += 1
                    return( self.coef_f_triangle_gauss(self.mesh[j], j, X_integ_i, IS-3 ) )
            
    def build_F(self) :

        self.F = np.array([ [ self.F_integ( i, j, self.X_integ[i]) for j in range(self.Ne)] for i in range(self.Ne)])
    




    ####### functions for symetric part and new mesh with same integration point

    def init_X_mesh_6pt_sym(self) :
        self.X_mesh_6pt_sym = np.array([ [ np.dot( self.N_6pt[i], self.mesh_sym[k] ) for i in range(6)] for k in range(self.Ne)])
        print( ' --- construction self.X_mesh_6pt_new shape : {}'.format(self.X_mesh_6pt_sym.shape) )


    def coef_f_gauss_6pt_sym(self, j, X_int) :
        return( np.dot( self.H_array(self.X_mesh_6pt_sym[j], X_int),(self.G_mesh_6pt[j]).T ) )


    def F_integ_sym(self, i, j, X_integ_i) : #, X_integ_4pt_i):

        d_ij = self.x_gap(X_integ_i, self.mesh_sym[j])

        if d_ij >= self.dmin :
            self.N_6pt_int += 1
            return( self.coef_f_gauss_6pt_sym(j, X_integ_i) )

        else :
            IS = self.IS_def(d_ij, self.size_elem[j] )
            if IS == 1 :
                self.N_6pt_int += 1
                return( self.coef_f_gauss_6pt_sym(j, X_integ_i) )

            elif IS < 5 :
                self.N_gauss_int += 1
                return( self.coef_f_gauss(self.mesh_sym[j], j, X_integ_i, IS ) )
            
            #self.N_gauss_int += 1
            #return( self.coef_f_gauss(self.mesh_sym[j], X_integ_i, 4 ) )   
            
            if IS == 5 :
                self.N_tri_int += 1
                return( self.coef_f_triangle_gauss(self.mesh_sym[j], j, X_integ_i, 3 ) )
            else : 
                self.N_tri_int += 1
                return( self.coef_f_triangle_gauss(self.mesh_sym[j], j, X_integ_i, IS-3 ) )
            


    def build_F_sym(self, mesh_sym) :

        self.init_mesh_sym(mesh_sym)
        self.init_X_mesh_6pt_sym()
        #return( np.array([ [ self.coef_f_gauss_6pt_sym( j, self.X_integ[i]) for j in range(self.Ne)] for i in range(self.Ne)]) )
        return( np.array([ [ self.F_integ_sym( i, j, self.X_integ[i]) for j in range(self.Ne)] for i in range(self.Ne)]) )
