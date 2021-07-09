#include <iostream>
#include <string.h>
#include <vector>
#include <H5Cpp.h>
#include "hdf5c.h"

using namespace std;
using namespace H5;

void hdf5c::hdf5_create_file(string filename) {
      hid_t hdf5_file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      herr_t status = H5Fclose(hdf5_file);
};

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

void hdf5c::hdf5_write_uint(string filename, string dataname, unsigned int data) {

  hid_t hdf5_file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t dataspace = H5Screate(H5S_SCALAR);
  hid_t dataset = H5Dcreate (hdf5_file, dataname.c_str(), H5T_NATIVE_UINT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  herr_t status;
  status = H5Dwrite(dataset, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);
  status = H5Dclose(dataset);
  status = H5Sclose(dataspace);
  status = H5Fclose(hdf5_file);
};

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

void hdf5c::hdf5_write_float(string filename, string dataname, float data) {

      hid_t hdf5_file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
      hid_t dataspace = H5Screate(H5S_SCALAR);
      hid_t dataset = H5Dcreate (hdf5_file, dataname.c_str(), H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      herr_t status;
      status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);
      status = H5Dclose(dataset);
      status = H5Sclose(dataspace);
      status = H5Fclose(hdf5_file);
};

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

void hdf5c::hdf5_write_double(string filename, string dataname, double data) {

  hid_t hdf5_file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t dataspace = H5Screate(H5S_SCALAR);
  hid_t dataset = H5Dcreate (hdf5_file, dataname.c_str(), H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  herr_t status;
  status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);
  status = H5Dclose(dataset);
  status = H5Sclose(dataspace);
  status = H5Fclose(hdf5_file);
};

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

void hdf5c::hdf5_write_array_float(string filename, string dataname, float* data, int dim_no, unsigned int* dims) {

  hsize_t dimsf[dim_no]; // dataset dimensions
  for(int ii = 0; ii < dim_no; ii++)
    dimsf[ii] = dims[ii];

  hid_t hdf5_file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t dataspace = H5Screate_simple(dim_no, dimsf, NULL);
  hid_t dataset = H5Dcreate (hdf5_file, dataname.c_str(), H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  herr_t status;
  status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  status = H5Dclose(dataset);
  status = H5Sclose(dataspace);
  status = H5Fclose(hdf5_file);
};

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

void hdf5c::hdf5_write_array_double(string filename, string dataname, double* data, int dim_no, unsigned int* dims) {

  hsize_t dimsf[dim_no]; // dataset dimensions
  for(int ii = 0; ii < dim_no; ii++)
    dimsf[ii] = dims[ii];

  hid_t hdf5_file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t dataspace = H5Screate_simple(dim_no, dimsf, NULL);
  hid_t dataset = H5Dcreate (hdf5_file, dataname.c_str(), H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  herr_t status;
  status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  status = H5Dclose(dataset);
  status = H5Sclose(dataspace);
  status = H5Fclose(hdf5_file);
};

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

void hdf5c::hdf5_write_vector_double(string filename, string dataname, vector<double> &data, unsigned int dim)
{

  hsize_t dimsf[1]; // dataset dimensions
  dimsf[0] = dim;

  hid_t hdf5_file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t dataspace = H5Screate_simple(1, dimsf, NULL);
  hid_t dataset = H5Dcreate (hdf5_file, dataname.c_str(), H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  herr_t status;
  status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]);
  status = H5Dclose(dataset);
  status = H5Sclose(dataspace);
  status = H5Fclose(hdf5_file);
}
