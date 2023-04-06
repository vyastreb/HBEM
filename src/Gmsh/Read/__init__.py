
import platform

sys_nd = platform.node()
if sys_nd == "cristal-login" :
    from .File4 import Geom_Gmsh_read4
    __all__ = ["File4"]

if sys_nd == "pbeguin.materiaux.ensmp.fr" :
    from .File import Geom_Gmsh_read 
    from .File4 import Geom_Gmsh_read4
    __all__ = ["File4", "File"]
