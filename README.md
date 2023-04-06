# HBEM: Fast-BEM method for Poisson's equation based on H-Matrices

+ Author: Paul Beguin
+ Contributor: Vladislav A. Yastrebov
+ Affiliation: MINES Paris, CNRS, France
+ License: BSD3

## Description

This is a Boundary Element Method solver based on the classical integration of singular integrals which uses hierarchical or H-matrices to accelerate the construction and resolution of the resulting linear system of equations. This code can read 2D meshes in gmsh format and solve the conductivity problem $\Delta U = 0$ with constant $U$ prescribed over the mesh, where $U$ can be interpreted either as an electric potential or a temperature.

## Content

The root contains the following folders
+ `src/` source code
+ `doc/` documentation
+ `tests/` testing facilities
+ `examples/` examples
  + `circular/` flux through a circular spot
  + `flower/` flux through a flower-shaped spot

## History

Apr 6 2023: First commit




