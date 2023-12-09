from setuptools import setup, Extension 
from Cython.Distutils import build_ext 
from Cython.Build import cythonize 


import numpy as np 

ext_options = {
    "language": "c++",
    "extra_compile_args": ["-O3"],
    "include_dirs": [np.get_include()]
}

extensions = [
    # Extension("shapemodule", sources=["shape_wrapper.pyx", "shape_computer.cpp"], **ext_options ),
    Extension("normmodule", sources=["np_wrapper.pyx", "norm_shape_computer.cpp"], **ext_options)
]

setup(ext_modules=cythonize(extensions))
