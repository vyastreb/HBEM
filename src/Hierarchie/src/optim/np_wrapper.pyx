cimport numpy as cnp 

cdef extern from "norm_shape_computer.hpp":
    double norm_cpp(double*)
    void norm_cpp_array(double * inp, double* out, const int& nbpt)
    int is_cpp( const double& d1, const double& d2 )
    void n_cpp( double * inp, double * out )
    void x_cpp( double * inp, double * inp_mesh, double * out )
    # void x_cpp_array(double ** inp_n, double * inp_mesh, double ** out, const int& nbpt )

cpdef norm_computer_optim( cnp.ndarray[cnp.float64_t] arr):
    return norm_cpp( <double*>arr.data )

cpdef norm_computer_array_optim( cnp.ndarray[cnp.float64_t] arr, cnp.ndarray[cnp.float64_t] out):
    norm_cpp_array( <double*>arr.data, <double*>out.data, out.size )

cpdef is_computer_optim( cnp.float64_t d1, cnp.float64_t d2 ):
    return is_cpp( <double>d1, <double>d2 )

cpdef n_computer_optim( cnp.ndarray[cnp.float64_t] inp, cnp.ndarray[cnp.float64_t] out):
    n_cpp( <double*>inp.data, <double*>out.data )

cpdef x_computer_optim( cnp.ndarray[cnp.float64_t] inp_n, cnp.ndarray[cnp.float64_t] inp_mesh, cnp.ndarray[cnp.float64_t] out):
    x_cpp( <double*>inp_n.data, <double*>inp_mesh.data, <double*>out.data )


