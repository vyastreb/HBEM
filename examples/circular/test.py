## import basic moduls
import numpy as np 
import os 
import time
import numpy.linalg as lin 

## matplotlib set up
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from matplotlib import cm
from matplotlib.colors import ListedColormap
mpl.use('agg')
# plt.style.use('my-style')

## module path gmsh and BEM
import sys
import platform 
sys.path.append( '../../src')

## modul for gmsh 
import Gmsh.Area as GmshArea
import Gmsh.Read as GmshRead

## modul for cluster
import Hierarchie.Cluster as Cluster
import Hierarchie.FastBem as Bem



#######################
##   Gmsh file read  ##
#######################

## input gmsh 
type_elem = 2
filename_msh = 'circ2.msh'
path_msh = '.'

mesh_Circ = GmshRead.Geom_Gmsh_read(type_elem, filename_msh, path_msh)
print( '\n --- read msh file name : %s \n'%(filename_msh) + 
    ' -     number of element %i \n\n'%( mesh_Circ.mesh_elem.shape[0] ) ) 



###############
##   Area    ##
###############

A = GmshArea.Area_mesh_tri_lin( mesh_Circ.mesh_elem )
print( ' --- area compute for the element mesh \n' + 
    ' -     total area %.3f \n -     error %.4f\n\n'%(np.sum(A), 1-np.sum(A)/np.pi) )



################
##  Cluster   ##
################

## input 
Sleaf = 5

circ_cluster = Cluster.Cluster(mesh_Circ, Sleaf)

Ncluster = len(circ_cluster.STotElem)
print( ' --- cluster formation \n' + 
    ' -     number clusters %i \n'%( len(circ_cluster.SLevel[0][0] ) ) + 
    ' -     size of cluster %i \n\n'%( len(circ_cluster.STotElem[0]) ) )
#    ' -     first cluster ' + str(circ_cluster.STotElem[0]) )



######################
##   Plot cluster   ##
# ####################

plt.figure(figsize=(6,6))

color_scale = cm.get_cmap('rainbow', Ncluster)

for i, C in enumerate(circ_cluster.STotElem) :
    for e in C :
        plt.fill(circ_cluster.mesh[e,:,0], circ_cluster.mesh[e,:,1], color=color_scale(i/Ncluster), alpha=0.9, lw=0.2)
        plt.plot(circ_cluster.mesh[e,:,0], circ_cluster.mesh[e,:,1], color='k', lw=0.2)

plt.axis('off')
plt.axis('equal')

os.chdir(path_msh)
plt.savefig('fig_Cluster.png', bbox_inches='tight')


## mesh, area vector rearrangement
STotElem = []
for S in circ_cluster.STotElem :
    STotElem += S
A_orga = np.array( [ A[s] for s in STotElem ] )
mesh_orga = np.array( [ mesh_Circ.mesh_elem[s] for s in STotElem ] )
Ne = len(A_orga)


####################
##  Integration   ##
####################

## input 
eta = 3 
err = 1e-4
err_bis = 1e-4
path_save = path_msh

print(' --- start of the integration with Fast Bem')
tfast_start = time.time()

## fast integration
Bem_circ = Bem.FastBem(circ_cluster, circ_cluster, 
    eta, err, err_bis, path_save, sing=True)

tfast_end = time.time()
print(' --- end of the integration \n -     time %.2f \n -     time block ACA %.2f \n -     time block sing %.2f \n'%(tfast_end - tfast_start, 
        Bem_circ.t_ACA, Bem_circ.t_block_sing) + 
    ' --- time integration \n -     time gauss 6pt %.2f \n -     time gauss %.2f \n -     time tri %.2f \n -     time sing %.2f \n\n'%(Bem_circ.t_6pt, 
        Bem_circ.t_gauss, Bem_circ.t_tri, Bem_circ.t_gauss) )


# # ### test algebra H_matrix
# from pprint import pprint

# C = np.zeros((8,8))

# A = np.array([[np.array([[np.eye(2),np.ones((2,2))], [np.ones((2,2)),np.eye(2)]], dtype=np.ndarray), np.ones((4,4))], 
#     [np.ones((4,4)), np.eye(4)]], dtype=np.ndarray)

# B = np.array([[np.eye(4), np.array([[np.eye(2),np.ones((2,2))], [np.ones((2,2)),np.eye(2)]], dtype=np.ndarray)], 
#     [np.ones((4,4)), np.eye(4)]], dtype=np.ndarray)

# DimSLevel = [[8], [4,4], [2,2,2,2]]
# Bem_circ.DimSLevel1 = DimSLevel
# Bem_circ.DimSLevel2 = DimSLevel
# Bem_circ.DimSLevel3_init(DimSLevel)

# # print(Bem_circ.Nlevel)
# Bem_circ.Nlevel = 0
# C = Bem_circ.ProductHmat(1, 0, A, B, C, 0, 0, 0)

# pprint(C)
# pprint( Bem_circ.AssembleBlockTree(C, 0, 0))

# pprint( np.allclose( np.dot(Bem_circ.AssembleBlockTree(A, 0, 0), Bem_circ.AssembleBlockTree(B, 0, 0) ) ,
#     Bem_circ.AssembleBlockTree(C, 0, 0) ) )


## assembly of the matrix
Bem_circ.Nlevel_a = 0
F_assemble = Bem_circ.AssembleBlockTree_AB(Bem_circ.Af_opti_tree, Bem_circ.Bf_opti_tree, 0, 0)


###################################
##   Comparison whitout fast BEM ##
###################################

from Hierarchie.FastBem_integ import Bem_integ

print( ' --- start of the full integration ')
tfull_start = time.time()

# cluster_d = { "STotElem":circ_cluster.STotElem, "STot":circ_cluster.STot, "SLevel":circ_cluster.SLevel, "SizeLevel":circ_cluster.SizeLevel, "mesh":circ_cluster.mesh }
Bem_integ_modul = Bem_integ(circ_cluster, circ_cluster, True)

F = np.zeros((Ne, Ne))
for i, sig in enumerate(STotElem) :
    for j, tau in enumerate(STotElem) :
        F[i,j] = Bem_integ_modul.F_integ_sing(sig, tau)

tfull_end = time.time()
print( ' --- end \n -     time %.2f \n -     error %.6f \n\n --- compression \n -    rate %i \n -    with optimisation %i \n\n'%( tfull_end - tfull_start, 
    lin.norm( (F_assemble-F)/F, 'fro'), 
    int(Bem_circ.CompCoefSafe*100/(Ne*Ne)), 
    int(Bem_circ.CompCoefSafeOpti*100/(Ne*Ne)) ) )

### plot rank matrix 
F_rank = Bem_circ.AssembleBlockTree_Frk(Bem_circ.F_opti_tree_rk, 0, 0, True)

## new color_scale 
color_scale = cm.get_cmap('plasma', 1000)
newcolors = color_scale(1 - np.float_power(np.linspace(0, 1, 1000), 1/3))
newcmp = ListedColormap(newcolors)

plt.figure(figsize=(6,7))

plt.imshow(F_rank, cmap=newcmp)
Bem_circ.PlotBlockTree(Bem_circ.F_opti_tree_rk, 0, 0, 0, 0)

### legend 
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = mpl.cm.ScalarMappable(cmap=newcmp, norm=norm)
sm.set_array([])

### ticks label
ticks_array = np.linspace(0, 1, 6)
labels = ['$ %i $'%(int(t*100)) for t in ticks_array ]

cbar = plt.colorbar(sm, ticks=ticks_array, orientation="horizontal", shrink=0.6)
cbar.set_label('$r_k \% $') #, fontsize=12)
cbar.ax.set_xticklabels( labels )

plt.axis('equal')
plt.axis('off')

### figname save
os.chdir(path_msh)
plt.savefig('Matrice_rk.png', bbox_inches='tight')



#####################
##      GMRES      ##
##   Resolution    ##
#####################

## import module
import scipy.sparse.linalg as lin_sc

Ne = len(A)

U = 0.5 * np.ones( Ne )
J = lin.solve(F, U)
t_g = time.time() # time mesuring
b = np.zeros( Ne )

# definition of matrice vector fonction
F_x = lambda x: Bem_circ.ProductVect_AB(1, 0,Bem_circ.Af_tree, Bem_circ.Bf_tree, x, b, 0, 0)
F_solve = lin_sc.LinearOperator( (Ne, Ne), F_x )

# gmres call
J_gmres = lin_sc.gmres( F_solve , U , tol=1e-8)[0]
t_g_end = time.time()

## resolution for neumann problem
J0 = np.ones( Ne )
U = np.zeros( Ne )
U = Bem_circ.ProductVect_AB(2, 0, Bem_circ.Af_tree, Bem_circ.Bf_tree, J0, U, 0, 0)
F = Bem_circ.AssembleBlockTree_AB( Bem_circ.Af_opti_tree, Bem_circ.Bf_opti_tree, 0, 0)
U2 = 2 * np.dot( F, J0 )

print( ' --- resolution process')
print( ' -     time gmres : %s '%(t_g_end - t_g ) )
print( ' -     accuracy |u - A.x|/|u| : %s ' % ( lin.norm( U - F_x(J_gmres)) / lin.norm( U ) ) )
print( ' -     global integ : %s' % (np.sum( A_orga*J_gmres ) ) )
print( ' -     global integ full : %s' % (np.sum( A_orga*J ) ) )
print( ' -     error : %s \n\n' % ( (np.sum(A_orga*J_gmres) - np.sum(A_orga*J)) / 4 ) )

print( ' --- resolution for Neumann problem j = 1')
print( ' -    resistance1 : %s'%( np.sum( A_orga * U ) / ( np.pi*np.pi ) ))
print( ' -    resistance2 : %s'%( np.sum( A_orga * U2 ) / ( np.pi*np.pi ) ))
print( ' -    resistance U0 : %s'%( 1 / np.sum( A_orga * J_gmres ) ) )


os.chdir(path_msh)
np.savez('Bem_result_J_A.npz', J=J_gmres, U=U, A=A_orga, mesh=mesh_orga)




##############################
##   Edition contact spot   ##
##############################


plt.figure(figsize=(6,8))

## load the results
os.chdir(path_msh)
file_save = np.load('Bem_result_J_A.npz', allow_pickle=True )
A, J, mesh = file_save['A'], file_save['J'], file_save['mesh']
file_save.close()


## fill mesh
color_scale = cm.get_cmap('viridis', 64)
Jmax, Jmin = 4, 0
for e, se in enumerate(STotElem) :
    mesh_e = mesh_Circ.mesh_elem[se]
    cs_e = color_scale( (J[e]-Jmin)/(Jmax-Jmin) )
    plt.fill(mesh_e[:,0], mesh_e[:,1], color=cs_e, alpha=0.9, lw=0.2)
    plt.plot(mesh_e[:,0], mesh_e[:,1], color='k', lw=0.2)


### legend 
norm = mpl.colors.Normalize(vmin=Jmin, vmax=Jmax)
sm = mpl.cm.ScalarMappable(cmap=color_scale, norm=norm)
sm.set_array([])


### ticks label
ticks_array = [Jmin, np.min(J), 1, 2, 3, Jmax ]
labels = [str(int(x)) if (type(x)==int) else '%.2f'%(x) for x in ticks_array ]


cbar = plt.colorbar(sm, ticks=ticks_array, orientation="horizontal", shrink=0.6)
cbar.set_label('$j_n$') #, fontsize=12)
cbar.ax.set_xticklabels( labels )


plt.axis('equal')
plt.axis('off')


### figname save
os.chdir(path_msh)
plt.savefig('Spot_jn.png', bbox_inches='tight', pad_inches=0)

del circ_cluster, Bem_circ, Bem_integ_modul
del F_rank, A, J, mesh
del A_orga, mesh_orga




####################################
##   Test with Neumann condition  ##
####################################


plt.figure(figsize=(6,8))

## load the results
os.chdir(path_msh)
file_save = np.load('Bem_result_J_A.npz', allow_pickle=True )
A, U, mesh = file_save['A'], file_save['U'], file_save['mesh']
file_save.close()


## fill mesh
color_scale = cm.get_cmap('viridis', 64)
Umax, Umin = np.max( U ), np.min( U )
for e, se in enumerate(STotElem) :
    mesh_e = mesh_Circ.mesh_elem[se]
    cs_e = color_scale( (U[e]-Umin)/(Umax-Umin) )
    plt.fill(mesh_e[:,0], mesh_e[:,1], color=cs_e, alpha=0.9, lw=0.2)
    plt.plot(mesh_e[:,0], mesh_e[:,1], color='k', lw=0.2)


# ### legend 
# norm = mpl.colors.Normalize(vmin=Jmin, vmax=Jmax)
# sm = mpl.cm.ScalarMappable(cmap=color_scale, norm=norm)
# sm.set_array([])


# ### ticks label
# ticks_array = [Jmin, np.min(J), 1, 2, 3, Jmax ]
# labels = [str(int(x)) if (type(x)==int) else '%.2f'%(x) for x in ticks_array ]


# cbar = plt.colorbar(sm, ticks=ticks_array, orientation="horizontal", shrink=0.6)
# cbar.set_label('$U$') #, fontsize=12)
# cbar.ax.set_xticklabels( labels )


plt.axis('equal')
plt.axis('off')


### figname save
os.chdir(path_msh)
plt.savefig('Spot_U.png', bbox_inches='tight', pad_inches=0)

del A, mesh, U
