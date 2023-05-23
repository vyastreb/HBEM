
import os 
import numpy as np

from .version2 import Geom_Gmsh_read2
from .version4 import Geom_Gmsh_read4

class Geom_Gmsh_read( Geom_Gmsh_read2, Geom_Gmsh_read4 ) :

    def __init__(self, type_elem, filename, path ) :

        self.type_elem = type_elem 
        self.filename = filename
        self.path = path 

        self.version_detec( )

        if self.version == 2 :
            Geom_Gmsh_read2.__init__( self, self.type_elem, self.filename, self.path )
        elif self.version == 4 :
            Geom_Gmsh_read4.__init__( self, self.type_elem, self.filename, self.path )
        

    def version_detec( self ) :

        ## filename
        os.chdir(self.path)
        file_r = open(self.filename, 'r')
        file_r2 = file_r.read()
        file_r.close()
        self.version = int( file_r2[12] )
        del file_r2


### program from FlowerAlpha
def MeshSRotate( mesh_init, S_init, dThe) :

    mesh_new = np.zeros((len(mesh_init), 3, 2))

    R_matrix = np.array([[np.cos(dThe), -np.sin(dThe)], [np.sin(dThe), np.cos(dThe)]])

    for k in range(len(mesh_init)) :
        mesh_line = np.zeros((3,2))
        for i in range(3) : ## linear
            mesh_line[i] = np.dot( R_matrix, mesh_init[k][i]) 
        mesh_new[k] = mesh_line


    S_new = []
    for k in range(len(S_init)) :
        STot_line = []
        for s_i in S_init[k] :
            STot_line.append( np.dot( R_matrix, s_i )  )
        S_new.append( np.array(STot_line) )
    
    return( mesh_new, S_new )



### program from FlowerAlpha
def MeshSMiror( mesh_init, S_init, dThe) :

    mesh_mirror = np.zeros((len(mesh_init), 3, 2))

    ## mirror
    R_mirror = np.array([[np.cos(2*dThe), np.sin(2*dThe)], [np.sin(2*dThe), -np.cos(2*dThe)]])
    for k in range(len(mesh_init)) :
        mesh_line = np.zeros((3,2))
        for i in range(3) : ## linear
            mesh_line[i] = np.dot( R_mirror, mesh_init[k][i]) 
        mesh_mirror[k] = mesh_line
    
    S_new = []
    for k in range(len(S_init)) :
        STot_line = []
        for s_i in S_init[k] :
            STot_line.append( np.dot( R_mirror, s_i )  )
        S_new.append( np.array(STot_line) )
    
    return( mesh_mirror, S_new )


