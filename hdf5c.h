#ifndef HDF5C_H
#define HDF5C_H
#include <string.h>
#include <vector>

using namespace std;

class hdf5c {
  public:
    void hdf5_create_file(string filename);
    // scalars
    void hdf5_write_uint(string filename, string dataname, unsigned int data);
    void hdf5_write_float(string filename, string dataname, float data);
    void hdf5_write_double(string filename, string dataname, double data);
    // arrays - multidimensional
    void hdf5_write_array_float(string filename, string dataname, float* data, int dim_no, unsigned int* dims);
    void hdf5_write_array_double(string filename, string dataname, double* data, int dim_no, unsigned int* dims);
    // vectors - 1d
    void hdf5_write_vector_double(string filename, string dataname, vector<double> &data, unsigned int dim);
};

#endif
