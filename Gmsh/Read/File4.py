
import os 
import numpy as np

from .tools import x_gap, elem_size_def


class Geom_Gmsh_read4 :


    def __init__(self, type_elem, filename, path) :

        self.filename_gmsh = filename
        self.path = path

        ### read gmsh and splitlines
        self.read_gmsh_file()

        ### nodes coordinates 
        self.ind_line = 1
        self.find_coord_section()
        self.type_element = type_elem
        
        ### elements mesh 
        if self.type_element in [2, 1] : # triangular lin
            self.elem_rk = [0, 1, 2]
        if self.type_element == 3 : # quadrangular lin
            self.elem_rk = [0, 1, 2, 3]
        if self.type_element == 9 : # triangular quad
            self.elem_rk = [0, 3, 1, 5, 4, 2]
        if self.type_element == 10 : # quadrangular quad
            self.elem_rk = [0, 4, 1, 7, 8, 5, 3, 6, 2]
        self.N_node_element = len(self.elem_rk)

        self.find_element_section()
        self.init_mesh_size()
        self.Ne = len(self.mesh_elem)
        print( self.Ne )
        print( self.mesh_elem[0] )
        self.X0_mesh = (1/3) * np.sum(self.mesh_elem, axis=1 )
        


    def read_gmsh_file(self) :

        os.chdir(self.path)
        file_r = open(self.filename_gmsh, 'r')
        file_r2 = file_r.read()
        file_r.close()
        file_list = file_r2.splitlines()
        self.file_list_str = file_list
        del file_r2, file_list

    def find_coord_section(self) :

        #self.coord_node = 
        str_start, str_end = '$Nodes', '$EndNodes'
        len_start, len_end = len(str_start), len(str_end)
        ind_coord = 1
        test, test2 = True, False

        coord_list = []
        while test : 
            line_str = self.file_list_str[self.ind_line]
            if test2 :
                if line_str[:len_end] == str_end:
                    test, test2 = False, False
                if self.ind_line == len(self.file_list_str)-1 :
                    test, test2 = False, False
            if test2 :
                #print( self.file_list_str[self.ind_line] )
                l = self.file_list_str[self.ind_line]
                n = int(l.split(' ')[3])- int(l.split(' ')[2]) 
                self.ind_line += n + 1 
                #print( self.file_list_str[self.ind_line] )
                #print( l )
                for k in range(n) :

                    l = self.file_list_str[self.ind_line]
                    #print( l )
                    x, y = l.split(' ')[0], l.split(' ')[1]
                    coord_list.append( [float(x), float(y)] )
                    ind_coord += 1
                    self.ind_line += 1

            if test2 == False : # start 
                if line_str[:len_start] == str_start:
                    test2 = True
                    ## skip the first line of the nodes section
                    self.ind_line += 1
                self.ind_line += 1
        
        del ind_coord, test, test2

        self.mesh_point = coord_list
        # print( self.mesh_point )
        print( len( self.mesh_point ) )
        del coord_list
    

    def find_element_section(self) :

        #self.coord_node = 
        str_start, str_end = '$Elements', '$EndElements'
        len_start, len_end = len(str_start), len(str_end)
        ind_elem = 1
        test, test2 = True, False

        elem_list = []
        
        while test : 
            line_str = self.file_list_str[self.ind_line]
            if test2 :
                if line_str[:len_end] == str_end:
                    test, test2 = False, False
                if self.ind_line == len(self.file_list_str)-1 :
                    test, test2 = False, False

            if test2 :
                word_list = line_str.split(' ')
                n = int(word_list[3])

                # print( word_list )
                if int(word_list[2]) == self.type_element :
                    self.ind_line += 1
                    
                    for _ in range(n) :
                        elem_list_b = []
                        line_k = self.file_list_str[self.ind_line].split(' ')
                        for i in self.elem_rk :
                            node_i = int(line_k[i+1])
                            # print( node_i )
                            elem_list_b.append(self.mesh_point[node_i-1]) #, self.mesh_point[node_i-1][2]])
                        elem_list.append( elem_list_b )
                        ind_elem += 1
                        self.ind_line += 1
                else :
                    self.ind_line += n+1
            
            if test2 == False : # start 
                if line_str[:len_start] == str_start:
                    test2 = True
                    self.ind_line += 1
                self.ind_line += 1
        
        del ind_elem, test, test2

        self.mesh_elem = np.array(elem_list)
        del elem_list
    

    def x_gap(self, X1, X2) :
        return( np.sqrt((X1[0]-X2[0])*(X1[0]-X2[0]) + (X1[1]-X2[1])*(X1[1]-X2[1])) )
        
    
    def ElemSize_def(self):

        ElemSize = np.array([0 for k in range(len(self.mesh_elem))] )
        for k in range(len(self.mesh_elem)) :
            elemk = self.mesh_elem[k]
            d13 = self.x_gap([elemk[1][1],elemk[1][2]], [elemk[9][1],elemk[9][2]])
            d24 = self.x_gap([elemk[3][1],elemk[3][1]], [elemk[7][1],elemk[7][2]])
            ElemSize[k] = np.max([d13, d24])
        self.ElemSize = ElemSize

    def __len__(self) :
        return( len(self.mesh_elem) )
    
    def init_mesh_size(self, ) :
        self.mesh_size = np.array([elem_size_def(mesh_k) for mesh_k in self.mesh_elem])



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