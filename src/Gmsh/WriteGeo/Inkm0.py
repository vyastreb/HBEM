


#############
## code for generation of fractal flower shape

### import module
import numpy as np 
import random as rd 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import os 
import shutil

import numpy.linalg as lin

mpl.use('agg')



class WriteGmshInkm0 :

    def __init__( self, param, path_pre, deref=False ) :

        self.param = param
        self.path_pre = path_pre
        self.deref = deref

        ### unfold dictionnary
        self.param_read()

        ### folder, path and readme file 
        self.folder_def()
        self.path = os.path.join( self.path_pre, self.folder_name )
        self.filename_gmsh_pre = 'Mesh_gmsh'
        self.index_document_def()
        self.Readme_write()


        ### filename gmsh
        self.figure_name = 'Ink_img.pdf'
        self.filename_option = 'Mesh_gmsh.msh.opt'
        self.filename_random = 'random{}.npz'.format( self.Nrd )

        ## ifilename gmsh 
        self.filename_gmsh = self.filename_gmsh_pre + str(self.ind_gmsh) + '.geo'
        self.filename_gmsh_opt = self.filename_gmsh_pre + str(self.ind_gmsh) + '.msh.opt'
        self.filename_param_fractal = 'fractal_param{}.npz'.format( self.ind_gmsh )

        ## filtering vectors and contour coordinates
        self.AK_filtering()
        self.Cont_init()

        ## find asperities and curvature
        self.alpha_init()
        self.Aspe_init()

        ## Spline definition
        self.IndSpline_def()

        ## Mesh size 
        self.h_mesh_daspe()

        ## Point deref
        if self.deref :
            self.pt_deref_init()
        else :
            self.N_deref = 0

        ## final gmsh file write
        self.write_gmsh_file()
        self.copy_file_gmsh_opt()

        ## plot of the geometry
        if self.figure_name not in os.listdir( self.path ) :
            self.figure_plot()


    ## read all parameters
    def param_read( self ):

        ## first param
        self.Nrd = self.param['Nrd']
        self.r0 = self.param['r0']
        self.m0 = self.param['m0']
        self.kl = self.param['kl']
        self.H = self.param['H']

        ## zeta parameter
        if 'zeta' in (self.param).keys() :
            self.zeta = self.param['zeta']
            self.ks = self.zeta * self.kl
        else :
            self.ks = self.param['N']
            self.zeta = (self.ks) // self.kl

        ## definition of r1
        A = np.sum( np.array( [(k/self.kl)**(-2*self.H-1) for k in range( self.kl, self.ks+1 ) ] ) )
        self.r1 = np.sqrt( 2*self.m0 / A )
        
        ## mesh dimension
        self.h0 = self.param['h0']
        self.h1 = self.param['h1']
        if 'h2' in (self.param).keys() :
            self.h2 = self.param['h2']
        else :
            self.h2 = self.h1
        
        ## regularisation 
        self.c = self.param['c']

        if 'alpha_mesh' in (self.param).keys() :
            self.alpha_mesh = self.param['alpha_mesh']
        else :
            self.alpha_mesh = 0.95 # default value 

        if 'r1_mesh' in (self.param).keys() :
            self.r1_mesh = self.param['r1_mesh']
        else :
            self.r1_mesh = self.r1*(self.kl)**(self.H+0.5)

        self.Ngmsh = (self.ks) * 32

        ## new directory for README file
        self.param_all = {'Nrd':self.Nrd, 'r0':self.r0, 'm0':self.m0, 'r1':self.r1, 'kl':self.kl, 'ks':self.ks, 'zeta':self.zeta, 'H':self.H, 'h0':self.h0, 'h1':self.h1, 'h2':self.h2, 'c':self.c, 'alpha_mesh':self.alpha_mesh, 'r1_mesh':self.r1_mesh, 'Ngmsh':self.Ngmsh }

    ## folder init
    ### composition of the filename
    def folder_def( self ) :

        ##
        Nrd_str = str(self.Nrd)
        kl_str = str(self.kl)
        ks_str = str(self.ks)
        m0_str = ''.join( (str(self.m0)).split('.') )
        H_str = ''.join( (str(self.H)).split('.') )

        self.folder_name = 'fractal_Nrd_'+Nrd_str+'_m0_'+m0_str+'_kl_'+kl_str+'_ks_'+ks_str+'_H_'+H_str+'_4'


        ### creating new folder
        if self.folder_name not in os.listdir( self.path_pre) :
            os.chdir( self.path_pre )
            os.mkdir( self.folder_name )
            print( ' --- create folder work :%s\n' % (self.folder_name) )

        
    ## definition of ind_gmsh
    def index_document_def( self ) :    
        ind_gmsh = 1
        os.chdir( self.path )
        for filename in os.listdir() :
            if (filename[:len(self.filename_gmsh_pre)] == self.filename_gmsh_pre) and ( 'geo' in filename ) :
                ind_gmsh += 1
        self.ind_gmsh = ind_gmsh
    
    ## Readme file
    def Readme_write( self ) :
        os.chdir( self.path )
        with open("README_{}.txt".format(self.ind_gmsh), 'w') as f: 
            for key, value in (self.param_all).items(): 
                f.write('%s:%s\n' % (key, value))

    ## init of the filtering vector and random orientation
    def AK_filtering( self ) :

        ## read the random orientation
        os.chdir( self.path_pre )
        self.the_rd = np.load( self.filename_random, allow_pickle=True )['the_rd'][:self.ks+1-self.kl]

        ## definition of the filtering vectors
        self.k_filter = np.array( [ k for k in range(self.kl, self.ks+1) ] )
        self.a_filter = np.array( [ self.r1*np.power( k/(self.kl), -(self.H+0.5)) for k in range(self.kl, self.ks+1)] )
        self.a1_filter = np.array( [ np.power( k/(self.kl), -self.H+0.5) for k in range(self.kl, self.ks+1)] )
        self.a2_filter = np.array( [ np.power( k/(self.kl), -self.H+1.5) for k in range(self.kl, self.ks+1)] )


    ## definition for r_plot 
    def h_def( self, the ) :
        return( np.sum( self.a_filter*np.cos( self.k_filter*the + self.the_rd) ) )

    
    ## unique rd list - same random radius and for angular
    def Cont_init( self ) :

        ## contour perturbation, regularisation and coorcinates
        self.the_plot = np.linspace(0, 2*(np.pi), self.Ngmsh+1 )[:self.Ngmsh]
        self.h_plot = np.array([self.h_def( t ) for t in self.the_plot] )
        self.r_plot = self.r0 * np.exp( self.h_plot )

        self.X = (self.r_plot)*np.cos( self.the_plot )
        self.Y = (self.r_plot)*np.sin( self.the_plot )

        ### save param fractal
        os.chdir( self.path )
        if self.filename_param_fractal not in os.listdir() :
            np.savez( self.filename_param_fractal, the_rd=self.the_rd, r_plot=self.r_plot, the_plot=self.the_plot )


    ### function for alpha definition
    def hp_def( self, the ) :
        return( -self.r1*self.kl*np.sum( self.a1_filter*np.sin( self.k_filter*the + self.the_rd) ) )

    def h1pp_def( self, the ) :
        return( -self.r1*self.kl*self.kl*np.sum( self.a2_filter*np.cos( self.k_filter*the + self.the_rd) ) )

    def h2pp_def( self, the ) :
        return( (self.r1*self.kl*np.sum( self.a1_filter*np.sin( self.k_filter*the + self.the_rd) ))**2 )

    ## function normalisation
    def f_c( self, x ) :
        return( (1-x)/(1+self.c*x) )
    

    def alpha_init( self ) :

        ## derivatives 
        self.rp_plot = np.array( [ self.hp_def( t ) for t in self.the_plot ] ) * self.r_plot
        self.r1pp_plot = np.array( [ self.h1pp_def( t ) for t in self.the_plot ] ) * self.r_plot
        self.r2pp_plot = np.array( [ self.h2pp_def( t ) for t in self.the_plot ] ) * self.r_plot
        self.rpp_plot = self.r1pp_plot + self.r2pp_plot

        ## curvature and normalisation
        self.alpha = (2* (self.rp_plot) + (self.r_plot)*(self.r_plot) - (self.r_plot)*(self.rpp_plot) ) / np.power( (self.rp_plot)*(self.rp_plot) + (self.r_plot)*(self.r_plot) , 3/2 )
        self.alpha_max = np.max( np.abs( self.alpha ) )
        self.alpha_norm = self.f_c( np.abs(self.alpha)/self.alpha_max )


    
    ## function find asperity index
    def Aspe_init( self ) :

        ## first detection - zero value algorithm
        R = []
        for k in range( self.Ngmsh ) :
            rk, rk1 = self.rp_plot[k], self.rp_plot[(k+1)%self.Ngmsh]
            if rk * rk1 < 0 :
                if min( abs(rk), abs(rk1) ) == abs(rk) :
                    R.append(k)
                else :
                    R.append((k+1)%self.Ngmsh)
        
        self.ind0 = R[0]
        self.Ind_aspe = [ i-self.ind0 for i in R ]
        self.Na = len( self.Ind_aspe )

        ## angle and radii of asperities
        self.Ta = [ self.the_plot[i] for i in R ]
        self.Ra = [ self.r_plot[i] for i in R ]

        ## new index asperities - angles radii for contour
        self.r_plot2 = [self.r_plot[(i+self.ind0)%self.Ngmsh] for i in range(self.Ngmsh)]
        self.the_plot2 = [self.the_plot[(i+self.ind0)%self.Ngmsh] for i in range(self.Ngmsh)]

        ## position X2, Y2
        self.X2 = self.r_plot2*np.cos( self.the_plot2 )
        self.Y2 = self.r_plot2*np.sin( self.the_plot2 )
        self.alpha_norm2 = np.array( [self.alpha_norm[(i+self.ind0)%self.Ngmsh] for i in range(self.Ngmsh) ] )


    def IndSpline_def( self ) :

        I = []

        ind0k, ind2k, k = self.Ind_aspe[0], self.Ind_aspe[1], 0
        while k < self.Na-1 :
            ind2k = self.Ind_aspe[k+1]
            if ind2k-ind0k > 5 :
                ind1k = (ind2k-ind0k)//2 + ind0k
                I.append( [k for k in range(ind0k, ind1k+1)] )
                I.append( [k for k in range(ind1k, ind2k+1)] )
                ind0k = ind2k
            elif ind2k-ind0k > 2 :
                I.append( [k for k in range(ind0k, ind2k+1)])
                ind0k = ind2k
            k += 1
            

        ## last segment 
        if self.Ngmsh-ind0k > 5 :
            ind1k = (self.Ngmsh-ind0k)//2 + ind0k
            I.append( [k for k in range(ind0k, ind1k+1)] )
            I.append( [k for k in range(ind1k, self.Ngmsh)] + [0] )
        else :
            I.append( [k for k in range(ind0k, self.Ngmsh)] + [0] )
    
        self.Ind_sp = I
        self.Nsp = len( self.Ind_sp )


    def h_mesh_daspe( self ) :

        ##
        self.h_mesh_1 = (self.h1-self.h2)*self.alpha_norm2 + self.h2
        
        h_mesh_init_2 = np.zeros( len(self.h_mesh_1) ) 
        for n in range( self.Nsp ):

            ## indices of the useful point
            n1, n2, n3 = self.Ind_sp[n-1][0], self.Ind_sp[n][0], self.Ind_sp[(n+1)%self.Nsp][0]
            x1, x2, x3 = self.X[n1], self.X[n2], self.X[n3]
            y1, y2, y3 = self.Y[n1], self.Y[n2], self.Y[n3]
            
            ## distance 
            d12 = np.sqrt( (x1-x2)**2 + (y1-y2)**2 )
            d23 = np.sqrt( (x2-x3)**2 + (y2-y3)**2 )
            d = min(d12, d23)

            ## new mesh 
            h_mesh_init_2[n2] = min( self.h_mesh_1[n2], (self.h1-self.h2)*d/(self.r1_mesh) + self.h2 )
        
        self.h_mesh_2 = h_mesh_init_2



    def pt_deref_init( self ) :

        ## select the radius and angle of cavity 
        r_cav, the_cav = [], []
        if self.rp_plot[0] > 0 : # first asperity a peak
            for i in range( (self.Na)//2 ) :
                r_cav.append( self.r_plot2[self.Ind_aspe[2*i+1]] )
                the_cav.append( self.the_plot2[self.Ind_aspe[2*i+1]] )
        else :
            for i in range( (self.Na)//2 ) :
                r_cav.append( self.r_plot2[self.Ind_aspe[2*i]] )
                the_cav.append( self.the_plot2[self.Ind_aspe[2*i]] )
        self.Rcav = r_cav
        self.Tcav = the_cav

        ## reset Xa position
        Xa = np.array( [ [self.Rcav[i]*np.cos(self.Tcav[i]), self.Rcav[i]*np.sin(self.Tcav[i])] for i in range( len( self.Rcav ) ) ] )

        ## position and mesh size of deref points
        R_deref, The_deref = [], []
        h_deref = []
        for k in range( self.Na ) :
            rk = (self.alpha_mesh)*(self.Rcav[k])
            #rk = 1 + np.log(0.9) - np.log(Ra_[k])
            Xk = np.array( [rk*np.cos(self.Tcav[k]), rk*np.sin(self.Tcav[k])] )

            dk = np.min( lin.norm(Xa - Xk, axis=0) )
            if dk > self.h2 :
                R_deref.append( rk )
                The_deref.append( self.Tcav[k] )
                h_deref.append( self.h2*dk/( self.r1_mesh ) )
        
        self.X_deref = np.array( R_deref ) * np.cos( np.array( The_deref ) )
        self.Y_deref = np.array( R_deref ) * np.sin( np.array( The_deref ) )
        self.h_deref = h_deref
        self.N_deref = len( self.X_deref )


    def write_gmsh_file( self ) :

        with open(self.filename_gmsh, 'w') as f: 

            ## point definition
            f.write( '\n')
            f.write( '// Points\n')
            f.write( 'Point(1) = {0, 0, 0, %s} ;\n' % (self.h0) )

            for k in range(self.Ngmsh) :
                xk, yk, hk = self.X2[k], self.Y2[k], self.h_mesh_2[k]
                f.write( 'Point(%s) = {%s , %s, 0, %s} ;\n' % (k+2, np.around(xk,decimals=5), np.around(yk,decimals=5), np.around(hk,decimals=5)))
            
            if self.deref :
                for k in range(self.N_deref) :
                    xk, yk, hk = self.X_deref[k], self.Y_deref[k], self.h_deref[k]
                    f.write( 'Point(%s) = {%s , %s, 0, %s} ;\n' % (self.Ngmsh+k+2, np.around(xk,decimals=5), np.around(yk,decimals=5), np.around(hk,decimals=5)))
                
            ## line definition
            ind_l = self.Ngmsh + self.N_deref + 2
            f.write( '\n')
            f.write( '// Spline\n')
            for k in range(self.Nsp) :
                f.write( 'Spline('+str(ind_l+k) + ') = {' + ', '.join( [str(ind_pt+2) for ind_pt in self.Ind_sp[k]]) + ' } ;\n'  )
            
            ## surface definition
            ind_s = self.Ngmsh + self.N_deref + self.Nsp + 2
            f.write( '\n')
            f.write( '// Line Loop and surface\n')
            f.write( 'Line Loop(' + str(ind_s) + ') = {' + ', '.join( [str(k) for k in range(ind_l, ind_s)] ) + ' } ;\n' )
            f.write( 'Plane Surface(%s) = { %s } ;\n' % ( ind_s+1, ind_s) )
            f.write( 'Point{1} In Surface{ %s } ;\n' % (ind_s+1) )
            
            ## deref point insert
            if self.deref :
                f.write( 'Point{' + ', '.join([ self.Ngmsh+k+2 for k in range(self.N_deref)]) + '} In Surface{ %s } ;\n' % (ind_s+1) )

        print( ' --- Write Mesh gmsh file : {}\n'.format( self.filename_gmsh ) )


    def copy_file_gmsh_opt( self ) :
        shutil.copy2( self.path_pre + '/' + self.filename_option, self.path + '/' + self.filename_gmsh_opt )
        print( ' --- Copy Mesh option file ' )



    def figure_plot( self ) :

        plt.figure(figsize=(10,10))

        plt.fill( 2*np.cos(self.the_plot), 2*np.sin(self.the_plot), color='cornflowerblue' )
        plt.fill( self.r_plot*np.cos( self.the_plot ), self.r_plot*np.sin( self.the_plot ), color='mediumblue')
        plt.plot( self.r_plot*np.cos( self.the_plot ), self.r_plot*np.sin( self.the_plot ), linewidth=2, color='navy')
        plt.plot( 1*np.cos(self.the_plot), 1*np.sin(self.the_plot), '--', linewidth=2, color='darkblue' )
        plt.text( 1.1/np.sqrt(2), 1.1/np.sqrt(2), '$r_{mean} = '+str(np.around(np.mean(self.r_plot),decimals=3))+'$', fontsize=20, color='darkblue')

        plt.axis([-2,2,-2,2])
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()

        os.chdir(self.path)
        plt.savefig(self.figure_name)

        print( ' --- Edition of the figure for the flower shape \n')





def WriteFlowerAlpha( hm, he, R_line1, R_line2, Coord_edge, dThe, h_line, filename ) :

    Line_gmsh = [' ']
    ind_pt = 1

    Nl, Ne = len( R_line1 ), len( Coord_edge[0] )

    Line_gmsh.append( 'Point('+str(ind_pt)+') = {0, 0, 0, '+str(hm)+'} ;')
    ind_pt += 1

    for k in range(1, Nl-1) :
        Line_gmsh.append( 'Point('+str(ind_pt)+') = {'+str(np.around(R_line1[k], decimals=6))+', 0, 0, '+str(np.around(h_line[k], decimals=6))+'} ;')
        ind_pt += 1


    for k in range(Ne) :
        Line_gmsh.append( 'Point('+str(ind_pt)+') = {'+str(np.around(Coord_edge[0][k], decimals=6))+' ,'+str(np.around(Coord_edge[1][k], decimals=6))+' ,0 ,'+str(he)+'} ;')
        ind_pt += 1
    

    for k in range(1,Nl-1) :
        Line_gmsh.append( 'Point('+str(ind_pt)+') = {'+str(np.around(R_line2[Nl-k-1]*np.cos(dThe), decimals=6))+',' +str(np.around(R_line2[Nl-k-1]*np.sin(dThe), decimals=6))+', 0, '+str(np.around(h_line[Nl-k-1], decimals=6))+'} ;')
        ind_pt += 1

    #Line_gmsh.append('Point('+str(ind_pt))
    Line_gmsh.append(' ')

    ind_line = 1
    for k in range( Nl-1 ) :
        Line_gmsh.append( 'Line('+str(ind_line)+') = {' + str(k+1) + ',' + str(k+2) + '} ;')
        ind_line += 1
    
    Line_gmsh.append( 'Spline(' + str(ind_line) + ') = {' + ' ,'.join( [str(k+1) for k in range(Nl-1, Nl+Ne-1)] ) + '} ;')
    ind_line += 1

    for k in range( Nl-2 ) :
        Line_gmsh.append( 'Line('+str(ind_line)+') = {' + str(Nl+Ne+k-1) + ',' + str(Nl+Ne+k) + '} ;')
        ind_line += 1

    Line_gmsh.append( 'Line('+str(ind_line)+') = {' + str(2*Nl+Ne-3) + ',1 } ;')
    Line_gmsh.append( ' ')

    Line_gmsh.append( 'Line Loop(' + str(ind_line+1) + ') = {' + ','.join([str(k+1) for k in range(ind_line) ]) + ' } ;' ) #1, 2, 3} ;')
    Line_gmsh.append( ' ' )

    Line_gmsh.append( 'Plane Surface(1) = {' + str(ind_line+1) + '} ;')

    str_gmsh = '\n'.join(Line_gmsh)
    
    file_w = open( filename, 'w')
    file_w.write( str_gmsh )
    file_w.close()

