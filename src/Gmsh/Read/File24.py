
import os 
import numpy as np


class Geom_Gmsh_read :


    def __init__(self, type_elem, path, filename) :

        self.filename_gmsh = filename
        self.path = path

        self.read_gmsh_file()
        self.version()
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

        self.ind_line = 0
        if self.v == 2 :
            self.find_coord_section()
            self.find_element_section()
        else : ## version 4
            self.find_coord_section4()
            self.find_element_section4()
        

    def read_gmsh_file(self) :

        os.chdir(self.path)
        file_r = open(self.filename_gmsh, 'r')
        file_r2 = file_r.read()
        file_r.close()
        file_list = file_r2.splitlines()
        self.file_list_str = file_list
        del file_r2, file_list

    ### define the version of the gmsh file
    def version(self) :
        self.v = int( self.file_list_str[1][0] )


    ### procedure version 2
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
                x, y = float(line_str.split(' ')[1]), float(line_str.split(' ')[2])
                coord_list.append( [ind_coord, x, y])
                ind_coord += 1
                del x, y

            if test2 == False : # start 
                if line_str[:len_start] == str_start:
                    test2 = True
                    self.ind_line += 1
            self.ind_line += 1
        
        del ind_coord, test, test2

        self.mesh_point = coord_list
        del coord_list
    

    def find_element_section(self) :

        #self.coord_node = 
        str_start, str_end = '$Elements', '$EndElements'
        len_start, len_end = len(str_start), len(str_end)
        ind_elem =  1
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
                if int(word_list[1]) == self.type_element :
                    elem_list.append([])

                    elem_list_b = []
                    for k in range(self.N_node_element) :
                        ind_pt_mesh = int(word_list[5 + k]) 
                        x, y = self.mesh_point[ind_pt_mesh-1][1], self.mesh_point[ind_pt_mesh-1][2]
                        elem_list_b.append([x, y])
                        del ind_pt_mesh, x, y

                    for k in range(self.N_node_element) : # adapted to the element geometry
                        elem_list[-1].append(elem_list_b[self.elem_rk[k]])
                    
                    del elem_list_b
                    ind_elem += 1
            
            if test2 == False : # start 
                if line_str[:len_start] == str_start:
                    test2 = True
                    self.ind_line += 1
            self.ind_line += 1
        
        del ind_elem, test, test2

        self.mesh_elem = np.array(elem_list)
        del elem_list


    ### procedure for version 4
    def find_coord_section4(self) :

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
                    coord_list.append( [ind_coord, float(x), float(y)] )
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
        del coord_list
    

    def find_element_section4(self) :

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

                if int(word_list[2]) == self.type_element :
                    self.ind_line += 1
                    
                    for k in range(n) :
                        elem_list_b = []
                        line_k = self.file_list_str[self.ind_line].split(' ')
                        for i in self.elem_rk :
                            node_i = int(line_k[i+1])
                            elem_list_b.append([self.mesh_point[node_i-1][1], self.mesh_point[node_i-1][2]])
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



