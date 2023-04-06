
#####################################
##  Modul, BEM module and H param  ##
#####################################

## import basic moduls
import numpy as np 
import os 
import numpy.linalg as lin 
import shutil
import time

## module path gmsh and BEM
import sys
# sys.path.append('/home/users02/pbeguin/BEM_modul_cristal')
sys.path.append( '/home/pbeguin/Bureau/BEM_modul')

## modul for gmsh 
import Gmsh.Read.File as GmshRead
import Gmsh.Area as GmshArea

## modul for H matrix
import H_modul_tri_lin_sym2

## BEM gmres resolution
import scipy.sparse.linalg as lin_sc


##################
##  Parameter   ##
##################

### location and filename
path = '/home/pbeguin/Bureau/BEM_modul_v2/examples/circular_v0'
filename_gmsh = 'circ2.msh'


### parameter for clustering
eta = 3 
err = 1e-4
err_b = 1e-4

### gmsh parameters
type_elem = 2


#######################
##   Gmsh file read  ##
#######################

Gmsh_flower = GmshRead.Geom_Gmsh_read(type_elem, path, filename_gmsh)
Ne = len(Gmsh_flower.mesh_elem)

## convertion of the mesh elem
mesh0 = np.zeros( (Ne,3,2) )
for k in range(Ne) :
    mesh0[k][0] = np.array([Gmsh_flower.mesh_elem[k][1][1], Gmsh_flower.mesh_elem[k][1][2] ] )
    mesh0[k][1] = np.array([Gmsh_flower.mesh_elem[k][2][1], Gmsh_flower.mesh_elem[k][2][2] ] )
    mesh0[k][2] = np.array([Gmsh_flower.mesh_elem[k][3][1], Gmsh_flower.mesh_elem[k][3][2] ] )


################
##   Cluster  ##
################

Cluster1 = H_modul_tri_lin_sym2.Cluster_tree( mesh0, 5 )

print( Cluster1.mesh[0] )
print( Cluster1.SLevel[0] )

Cluster1_d = { "STotElem":Cluster1.STotElem, "STot":Cluster1.STot, "SLevel":Cluster1.SLevel, "SizeLevel":Cluster1.SizeLevel, "mesh":Cluster1.mesh }

print( ' --- cluster formation \n' + 
    ' -     number clusters %i \n'%( len(Cluster1.SLevel[0][0] ) ) + 
    ' -     size of cluster %i \n'%( len(Cluster1.STotElem[0]) )  + 
    ' -     first cluster ' + str(Cluster1.STotElem[0]) )

#####################
##  Area ordening  ##
#####################

Area = GmshArea.Area_mesh_tri_lin( Gmsh_flower.mesh_elem )

STotElem = []
for S in Cluster1.STotElem :
    STotElem += S

print( '\n\n --- Area vector\n ----- N_cluster : {} \n ----- shape S : {}'.format( len(Cluster1.STotElem), len(STotElem) ) )

A_orga = np.array( [ Area.A[s-1] for s in STotElem ] )
Mesh_orga = np.array( [ mesh0[s-1] for s in STotElem ] )


##############
##   BEM    ##
##############
"""
BEM_11 = H_modul_tri_lin_sym2.FMM_mesh_sym( Cluster1_d, Cluster1_d, eta, err, err_b, path, sing = True )

## assemble matrix 
BEM_11.Nlevel_a = 0
F_assemble = BEM_11.AssembleBlockTree_AB( BEM_11.AF_tree, BEM_11.BF_tree, 0, 0)

## tempary saving
os.chdir( path )
np.savez( 'BEM_result_temp.npz', F_tree_adm = BEM_11.F_tree_adm, F_tree_rk = BEM_11.F_tree_rk, 
    F_opti_tree_rk =BEM_11.F_opti_tree_rk, AF_tree = BEM_11.AF_tree, BF_tree = BEM_11.BF_tree, 
    AF_opti_tree = BEM_11.AF_opti_tree, BF_opti_tree = BEM_11.BF_opti_tree, 
    F_assemble = F_assemble)


##############
##   GMRES  ##
##############

U = 0.5 * np.ones( Ne )
t_g = time.time() # time mesuring
b = np.zeros( Ne )

## definition of matrice vector fonction
F_x = lambda x: BEM_11.ProductVect_AB(1, 0, BEM_11.AF_tree, BEM_11.BF_tree, x, b, 0, 0)
F_solve = lin_sc.LinearOperator( (Ne, Ne), F_x )

## gmres call
J_gmres = lin_sc.gmres( F_solve , U , tol=1e-8)[0]
t_g_end = time.time()

print( ' -- time gmres : %s '%(t_g_end - t_g ) )
print( ' -- accuracy |u - A.x|/|u| : %s ' % ( lin.norm( U - F_x(J_gmres)) / lin.norm( U ) ) )
print( ' -- global integ : %s\n\n' % (np.sum( A_orga*J_gmres ) ) )


## comparison with initial matrix
J = lin.solve( F_assemble, U )
print( ' -- global integ ref : %s\n\n' % (np.sum( A_orga*J ) ) )


## save 
os.chdir( path )
np.savez('BEM_result_J_A.npz', J=J_gmres, A=A_orga, mesh=Mesh_orga, 
    STotElem=STotElem, NCoef=BEM_11.CompCoef, NCoefSave=BEM_11.CompCoefSafe, NCoefOpti=BEM_11.CompCoefSafeOpti )


"""

