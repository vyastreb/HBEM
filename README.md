# :rainbow: HBEM: Fast-BEM method for Poisson's equation based on H-Matrices

+ Author: Paul Beguin
+ Contributor: Vladislav A. Yastrebov
+ Affiliation: MINES Paris, CNRS, France :fr:
+ License: BSD3 :unlock:

## :book: Description

This is a Boundary Element Method solver based on the classical integration of singular integrals which uses hierarchical or H-matrices to accelerate the construction and resolution of the resulting linear system of equations. This code can read 2D meshes in gmsh format and solve the conductivity problem $\Delta U = 0$ over flat half-space $z=0$ with constant $U=U_0$ prescribed at the mesh $\Omega$, the remaining space $\mathbb R^2\setminus\Omega$ is assumed to be isolated $\partial U/\partial z = 0$, where $U$ can be interpreted either as an electric potential or a temperature. Constant interpolation of the flux through the element is used.

## :green_book: Content

The root contains the following folders
+ `src/` source code
  + `mesh_reader.py`
  + `solver.py`
  + `plot.py`
+ `doc/` documentation
+ `tests/` testing facilities
+ `tools/` additional tools (`GridToBezier.py`)
+ `examples/` examples
  + `circular/` flux through a circular spot
  + `flower/` flux through a flower-shaped spot

## TODO

+ @PaulBegiun Clean up the reader that it does not detect any specific platform, if needed to use different gmsh versions, it need to be specified as function's argument or could be detected on flight from *.msh file.


## :paperclip: History

+ 9 Dec 2023: added GridToBezier tool
+ 6 Apr 2023: First commit





