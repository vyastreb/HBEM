

## import modul
import os

import sys 
sys.path.append('/home/pbeguin/Bureau/BEM_modul')

import GmshReadFile_modul4 as gm 


path_file = '/home/pbeguin/Bureau/git_Papers/git_workspace/TASKS/Zset_contact_spot/fractal_nauti_spot/fractal_kl_4/fractal_Nrd_1_r1_005_kl_4_ks_128_H_025_4'

file_name = 'Mesh_gmsh1'
file_name2 = 'Mesh_gmsh1'

gm_file = gm.Gmsh_convert(path_file, file_name, file_name2, [2], [4421])
print( gm_file.lines_elem[0] )
gm_file.convert_v2()
