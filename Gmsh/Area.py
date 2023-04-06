
import numpy as np
import numpy.linalg as lin

def Area_mesh_tri_lin(Mesh) : 

	""" function which returns an array of mesh area element
		-- for linear triangular element mesh
	"""
	G_2D = np.array([[-1, 1, 0], [-1, 0, 1]])
	return( np.array([ 0.5*lin.det(np.dot(G_2D, mesh_k)) for mesh_k in Mesh]) )
        
