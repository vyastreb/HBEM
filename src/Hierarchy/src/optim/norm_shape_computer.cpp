#include <cmath>
#include <algorithm>
#include <vector>

double norm_cpp(double* arr){
    return sqrt( pow(arr[0], 2) + pow(arr[1], 2));
}

void norm_cpp_array(double * inp, double* out, const int& nbpt){
    for( int i=0; i<nbpt; i++){
        out[i] = 1./sqrt( pow(inp[2*i], 2) + pow(inp[2*i+1], 2));
    }
}

int is_cpp( const double& d1, const double& d2 ){
    int i ;
    i = floor( 2.37 * d2 / d1 ) - 1 ;
    return std::min( std::max( 1, i ), 8 );
}

void n_cpp(double* inp, double* out ){
    out[0] = 1. - inp[0] - inp[1] ;
    out[1] = inp[0] ;
    out[2] = inp[1] ;
}

std::vector<double> n_x_cpp(double* inp ){
    std::vector<double> out(3) ;
    out[0] = 1. - inp[0] - inp[1] ;
    out[1] = inp[0] ;
    out[2] = inp[1] ;
    return out ;
}

void x_cpp(double * inp_n, double * inp_mesh, double* out ){
    
    std::vector<double> n(3) ;
    n = n_x_cpp( inp_n ) ;

    out[0] = inp_mesh[0]*n[0] + inp_mesh[2]*n[1] + inp_mesh[4]*n[2] ;
    out[1] = inp_mesh[1]*n[0] + inp_mesh[3]*n[1] + inp_mesh[5]*n[2] ;
}
