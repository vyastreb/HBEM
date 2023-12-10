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
    Extension("gaussmodule", sources=["read_gausspt.pyx"], **ext_options)
]

setup(ext_modules=cythonize(extensions))