
import os 
import numpy as np

from .tools import x_gap, elem_size_def

class Geom_Gmsh_read2( object ) :


    def __init__(self, type_elem, filename, path) :

        self.filename_gmsh = filename
        self.path = path

        self.read_gmsh_file()
        self.find_coord_section()
        self.type_element = type_elem
        
        if self.type_element == 2 : # triangular lin
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
        ind_line, ind_coord = 0, 1
        test, test2 = True, False

        coord_list = []
        while test : 
            line_str = self.file_list_str[ind_line]
            if test2 :
                if line_str[:len_end] == str_end:
                    test, test2 = False, False
                if ind_line == len(self.file_list_str)-1 :
                    test, test2 = False, False
            if test2 :
                x, y = float(line_str.split(' ')[1]), float(line_str.split(' ')[2])
                coord_list.append( [ind_coord, x, y])
                ind_coord += 1
                del x, y

            if test2 == False : # start 
                if line_str[:len_start] == str_start:
                    test2 = True
                    ind_line += 1
            ind_line += 1
        
        del ind_line, ind_coord, test, test2

        self.mesh_point = coord_list
        del coord_list
    

    def find_element_section(self) :

        #self.coord_node = 
        str_start, str_end = '$Elements', '$EndElements'
        len_start, len_end = len(str_start), len(str_end)
        ind_line, ind_elem = 0, 1
        test, test2 = True, False

        elem_list = []
        
        while test : 
            line_str = self.file_list_str[ind_line]
            if test2 :
                if line_str[:len_end] == str_end:
                    test, test2 = False, False
                if ind_line == len(self.file_list_str)-1 :
                    test, test2 = False, False

            if test2 :
                word_list = line_str.split(' ')
                if int(word_list[1]) == self.type_element :
                    elem_list.append([])

                    elem_list_b = []
                    for k in range(self.N_node_element) :
                        ind_pt_mesh = int(word_list[5 + k]) 
                        x, y = self.mesh_point[ind_pt_mesh-1][1], self.mesh_point[ind_pt_mesh-1][2]
                        elem_list_b.append([x, y])
                        del x, y

                    for k in range(self.N_node_element) : # adapted to the element geometry
                        elem_list[-1].append(elem_list_b[self.elem_rk[k]])
                    
                    del elem_list_b
                    ind_elem += 1
            
            if test2 == False : # start 
                if line_str[:len_start] == str_start:
                    test2 = True
                    ind_line += 1
            ind_line += 1
        
        del ind_line, ind_elem, test, test2

        self.mesh_elem = np.array(elem_list)
        del elem_list
    

    def init_mesh_size(self, ) :
        self.mesh_size = np.array([elem_size_def(mesh_k) for mesh_k in self.mesh_elem])

    