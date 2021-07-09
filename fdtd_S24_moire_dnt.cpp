#include <iostream>
#include <complex>
#include <math.h>
#include <omp.h>
#include <chrono>
#include "hdf5c.h"

using namespace std;

// Finite difference time domain simulation for trapping a light pulse in a dynamic morie grating

double sellmeier_fused_silica(double l0);
void ms_format(double ms);

int main(void)
{

  unsigned int nthreads = omp_get_max_threads();

  // set hfd5 class for outputing data
  hdf5c h5;
  string filename = "fdtd_data.h5";

  // imaginary unit
  complex<double> i (0.0, 1.0);

  // constants
  double c = 299792458;
  double e0 = 8.85418782e-12;
  double u0 = 1.25663706e-6;
  double nu = sqrt(u0 / e0);
  double pi = 3.14159265359;

  // interators
  unsigned int ii;
  unsigned int jj;
  unsigned int kk;

  ///////////////////////////////////////////
  ///////////////// inputs //////////////////
  ///////////////////////////////////////////

  ///////////////////////////////////////////
  unsigned int bw_nopo = 1501;
  unsigned int tv_nopo_input = 1000;
  unsigned int xv_nopo_input = 1000;
  unsigned int res = 10;
  ///////////////////////////////////////////
  double l0 = 1.55e-06;
  double pfbw = 1e12;
  ///////////////////////////////////////////
  double glen = 0.004;
  double gstart = 1e-4;
  double dn = 1e-1;
  double ds = 2.4e-05;
  double gramp_scale = 0.35;
  ///////////////////////////////////////////
  double tlen = 60e-12;
  double mu = 4;
  double tstart = 18e-12;
  double delta_t = 5e-12;
  double ton = 10e-12;
  ///////////////////////////////////////////

  // ensure bw_nopo is an odd number
  if (bw_nopo % 2 == 0)
    bw_nopo += 1;

  ///////////////////////////////////////////
  ////////////// write inputs ///////////////
  ///////////////////////////////////////////

  h5.hdf5_create_file(filename);
  h5.hdf5_write_uint(filename, "bw_nopo", bw_nopo);
  h5.hdf5_write_uint(filename, "tv_nopo_input", tv_nopo_input);
  h5.hdf5_write_uint(filename, "xv_nopo_input", xv_nopo_input);
  h5.hdf5_write_uint(filename, "res", res);
  h5.hdf5_write_double(filename, "l0", l0);
  h5.hdf5_write_double(filename, "pfbw", pfbw);
  h5.hdf5_write_double(filename, "glen", glen);
  h5.hdf5_write_double(filename, "gstart", gstart);
  h5.hdf5_write_double(filename, "dn", dn);
  h5.hdf5_write_double(filename, "ds", ds);
  h5.hdf5_write_double(filename, "gramp_scale", gramp_scale);
  h5.hdf5_write_double(filename, "tlen", tlen);
  h5.hdf5_write_double(filename, "mu", mu);
  h5.hdf5_write_double(filename, "tstart", tstart);
  h5.hdf5_write_double(filename, "delta_t", delta_t);
  h5.hdf5_write_double(filename, "ton", ton);

  ///////////////////////////////////////////
  ///////////////////////////////////////////
  ///////////////////////////////////////////

  // set pulse carrier parameters
  double k0 = (2 * pi) / l0; // wavenumber
  double w0 = c * k0; // angular frequency
  double f0 = w0 / (2 * pi); // frequency
  double n0 = sellmeier_fused_silica(l0); // set refractive index
  double vp0 = c / n0; // phase velocity
  double beta0 = n0 * k0; // propagation constant

  // Bragg grating period (m)
  double dB = l0 / (2 * n0);

  h5.hdf5_write_double(filename, "n0", n0);

  ///////////////////////////////////////////
  ///////// set bandwidth and pulse /////////
  ///////////////////////////////////////////

  double fbw = 1.5* pfbw;
  int width_div = 100;
  double sigma = (pi * pfbw) / (c * sqrt(2*log(width_div)));

  vector<double> kv(bw_nopo);
  vector<double> wv(bw_nopo);
  vector<double> fk(bw_nopo);
  double fk_sum = 0;

  for (ii = 0; ii < bw_nopo; ii++)
  {
    kv[bw_nopo-ii-1] = (pi * fbw) / c * ((double)(2 * ii) / (bw_nopo - 1) - 1) + k0;
    wv[bw_nopo-ii-1] = c * kv[bw_nopo-ii-1];
    fk[ii] = exp(-0.5*pow((kv[bw_nopo-ii-1]-k0)/sigma, 2));
    fk_sum = fk[ii];
  }

  // normalise fk
  for (ii = 0; ii < bw_nopo; ii++)
    fk[ii] = fk[ii]/fk_sum;

  h5.hdf5_write_vector_double(filename, "kv", kv, bw_nopo);
  h5.hdf5_write_vector_double(filename, "fk", fk, bw_nopo);

  double FWHM = (c * sigma * sqrt(2*log(2))) / pi;
  double pulse_tlen = (2 / (c*sigma)) * sqrt(2*log(width_div));
  double pulse_xlen = pulse_tlen * vp0;

  ///////////////////////////////////////////
  /////////////// set postions //////////////
  ///////////////////////////////////////////

  double xlen = 2 * gstart + glen;
  double gramp = gramp_scale * glen / 2;

  double dx_calc_bare = dB / res;
  unsigned int xv_nopo_calc = floor(xlen / dx_calc_bare) + 1;

  double gfstart = gstart + gramp;
  double gfend = xlen - gstart - gramp;
  double gend = xlen - gstart;

  vector<double> xv_calc(xv_nopo_calc);
  vector<double> apod_calc(xv_nopo_calc);
  vector<double> gm_calc(xv_nopo_calc);

  #pragma omp parallel for
  for (ii = 0; ii < xv_nopo_calc; ii++)
  {
    xv_calc[ii] = (xlen * ii) / (xv_nopo_calc - 1);
    if (xv_calc[ii] >= gstart && xv_calc[ii] <= gfstart)
      apod_calc[ii] = 0.5 + 0.5*cos((pi/(gfstart-gstart))*(xv_calc[ii]-gfstart));
    else if (xv_calc[ii] > gfstart && xv_calc[ii] < gfend)
      apod_calc[ii] = 1;
    else if (xv_calc[ii] >= gfend && xv_calc[ii] <= gend)
      apod_calc[ii] = 0.5 + 0.5*cos((pi/(gend-gfend))*(xv_calc[ii]-gfend));
    gm_calc[ii] = apod_calc[ii] * cos((2*pi/ds)*xv_calc[ii]) * cos((2*pi/dB)*xv_calc[ii]);
  }

  double dx_calc = xv_calc[1]-xv_calc[0];
  unsigned int xv_src_idx = 4;
  double xv_src = xv_calc[xv_src_idx];

  ///////////////////////////////////////////
  //////////// postions reporting ///////////
  ///////////////////////////////////////////

  unsigned int xv_sep = floor(xv_nopo_calc / xv_nopo_input) + 1;
  unsigned int xv_nopo = xv_nopo_input;

  for (ii = xv_nopo_input; ii > 0; ii--)
  {
    if ((ii-1)*xv_sep < xv_nopo_calc)
    {
      xv_nopo = ii;
      break;
    }
  }

  vector<double> xv(xv_nopo);
  vector<double> apod(xv_nopo);
  vector<double> gm(xv_nopo);

  for (ii = 0; ii < xv_nopo; ii++)
  {
    xv[ii] = xv_calc[xv_sep*ii];
    apod[ii] = apod_calc[xv_sep*ii];
    gm[ii] = gm_calc[xv_sep*ii];
  }

  h5.hdf5_write_double(filename, "gfstart", gfstart);
  h5.hdf5_write_double(filename, "gfend", gfend);
  h5.hdf5_write_double(filename, "gend", gend);
  h5.hdf5_write_uint(filename, "xv_nopo", xv_nopo);
  h5.hdf5_write_vector_double(filename, "xv", xv, xv_nopo);
  h5.hdf5_write_vector_double(filename, "apod", apod, xv_nopo);
  h5.hdf5_write_vector_double(filename, "gm", gm, xv_nopo);

  ///////////////////////////////////////////
  //////////////// set timings //////////////
  ///////////////////////////////////////////

  double t_delay = 0.9 * (n0 * pulse_xlen) / c;
  double dt_calc = (n0 * dx_calc) / (2*c);

  unsigned int tv_nopo_calc = int(tlen / dt_calc) + 1;
  tlen = (tv_nopo_calc - 1) * dt_calc;

  vector<double> tv_calc(tv_nopo_calc);
  vector<double> twindow_calc(tv_nopo_calc);
  complex<double>* source_calc = new complex<double>[tv_nopo_calc];

  #pragma omp parallel for private(jj)
  for (ii = 0; ii < tv_nopo_calc; ii++)
  {
    tv_calc[ii] = ii *  dt_calc;

    if (tv_calc[ii] >= tstart && tv_calc[ii] <= tstart + delta_t)
      twindow_calc[ii] = mu*(0.5 + 0.5*cos((pi/delta_t)*(tv_calc[ii]-tstart-delta_t)));
    else if (tv_calc[ii] > tstart + delta_t && tv_calc[ii] < tstart + delta_t + ton)
      twindow_calc[ii] = mu;
    else if (tv_calc[ii] >= tstart + delta_t + ton && tv_calc[ii] <= tstart + 2*delta_t + ton)
      twindow_calc[ii] = mu*(0.5 + 0.5*cos((pi/delta_t)*(tv_calc[ii]-tstart-delta_t-ton)));

    for (jj = 0; jj < bw_nopo; jj++)
        source_calc[ii] = source_calc[ii] + fk[jj] * i * exp(i*wv[jj]*(tv_calc[ii] - t_delay));
  }

  ///////////////////////////////////////////
  ///////////// timing reporting ////////////
  ///////////////////////////////////////////

  unsigned int tv_sep = floor(tv_nopo_calc / tv_nopo_input) + 1;
  unsigned int tv_nopo = tv_nopo_input;

  for (ii = tv_nopo_input; ii > 0; ii--)
  {
    if ((ii-1)*tv_sep < tv_nopo_calc)
    {
      tv_nopo = ii;
      break;
    }
  }

  vector<double> tv(tv_nopo);
  vector<double> twindow(tv_nopo);
  vector<double> source_real(tv_nopo);
  vector<double> source_imag(tv_nopo);

  for (ii = 0; ii < tv_nopo; ii++)
  {
    tv[ii] = tv_calc[tv_sep*ii];
    twindow[ii] = twindow_calc[tv_sep*ii];
    source_real[ii] = source_calc[tv_sep*ii].real();
    source_imag[ii] = source_calc[tv_sep*ii].imag();
  }

  h5.hdf5_write_uint(filename, "tv_nopo", tv_nopo);
  h5.hdf5_write_vector_double(filename, "tv", tv, tv_nopo);
  h5.hdf5_write_vector_double(filename, "twindow", twindow, tv_nopo);
  h5.hdf5_write_vector_double(filename, "source_real", source_real, tv_nopo);
  h5.hdf5_write_vector_double(filename, "source_imag", source_imag, tv_nopo);

  ///////////////////////////////////////////
  ////////////// run simulation /////////////
  ///////////////////////////////////////////

  double m =  (c * dt_calc) / (24*dx_calc);

  complex<double>* efield = new complex<double>[xv_nopo_calc];
  complex<double>* dfield = new complex<double>[xv_nopo_calc];
  complex<double>* hfield = new complex<double>[xv_nopo_calc];

  complex<double>* efield_lbc = new complex<double>[3];
  complex<double>* efield_ubc = new complex<double>[5];
  complex<double>* hfield_lbc = new complex<double>[5];
  complex<double>* hfield_ubc = new complex<double>[3];

  vector<double> efield_mat_real(tv_nopo*xv_nopo);
  vector<double> efield_mat_imag(tv_nopo*xv_nopo);
  vector<double> hfield_mat_real(tv_nopo*xv_nopo);
  vector<double> hfield_mat_imag(tv_nopo*xv_nopo);

  auto now = std::chrono::high_resolution_clock::now();
  auto total_timing = now;
  auto timing = total_timing;

  for (ii = 0; ii < tv_nopo-1; ii++)
  {

    for (jj = tv_sep*ii+1; jj <= tv_sep*(ii+1); jj++)
    {

      ///////////////////////////////////////////////////////////////
      ///////////////// update H field from E field /////////////////
      ///////////////////////////////////////////////////////////////

      hfield[0] = hfield[0] + m * (-efield[2] + 27.0*efield[1] - 27.0*efield[0] + efield_lbc[2]);
      #pragma omp parallel for private(kk)
      for (kk = 1; kk < xv_nopo_calc-2; kk++)
        hfield[kk] = hfield[kk] + m * (-efield[kk+2] + 27.0*efield[kk+1] - 27.0*efield[kk] + efield[kk-1]);
      hfield[xv_nopo_calc-2] = hfield[xv_nopo_calc-2] + m * (-efield_ubc[2] + 27.0*efield[xv_nopo_calc-1] - 27.0*efield[xv_nopo_calc-2] + efield[xv_nopo_calc-3]);
      hfield[xv_nopo_calc-1] = hfield[xv_nopo_calc-1] + m * (-efield_ubc[4] + 27.0*efield_ubc[2] - 27.0*efield[xv_nopo_calc-1] + efield[xv_nopo_calc-2]);

      // store H field lower boundaries
      hfield_lbc[4] = hfield_lbc[3];
      hfield_lbc[3] = hfield_lbc[2];
      hfield_lbc[2] = hfield_lbc[1];
      hfield_lbc[1] = hfield_lbc[0];
      hfield_lbc[0] = hfield[0];

      // store H field upper boundaries
      hfield_ubc[2] = hfield_ubc[1];
      hfield_ubc[1] = hfield_ubc[0];
      hfield_ubc[0] = hfield[xv_nopo_calc-1];

      ///////////////////////////////////////////////////////////////
      ///////////////// update D field from H field /////////////////
      ///////////////////////////////////////////////////////////////

      dfield[0] = dfield[0] + m * (-hfield[1] + 27.0*hfield[0] - 27.0*hfield_lbc[2] + hfield_lbc[4]);
      dfield[1] = dfield[1] + m * (-hfield[2] + 27.0*hfield[1] - 27.0*hfield[0] + hfield_lbc[2]);
      #pragma omp parallel for private(kk)
      for (kk = 2; kk < xv_nopo_calc-1; kk++)
        dfield[kk] = dfield[kk] + m * (-hfield[kk+1] + 27.0*hfield[kk] - 27.0*hfield[kk-1] + hfield[kk-2]);
      dfield[xv_nopo_calc-1] = dfield[xv_nopo_calc-1] + m * (-hfield_ubc[2] + 27.0*hfield[xv_nopo_calc-1] - 27.0*hfield[xv_nopo_calc-2] + hfield[xv_nopo_calc-3]);

      ///////////////////////////////////////////////////////////////
      ///////////////// update E field from D field /////////////////
      ///////////////////////////////////////////////////////////////

      #pragma omp parallel for private(kk)
      for (kk = 0; kk < xv_nopo_calc; kk++)
        efield[kk] = dfield[kk] / pow(n0 + dn*(1 + twindow_calc[jj])*gm_calc[kk], 2);

      // store E field lower boundaries
      efield_lbc[2] = efield_lbc[1];
      efield_lbc[1] = efield_lbc[0];
      efield_lbc[0] = efield[0];

      // store E field upper boundaries
      efield_ubc[4] = efield_ubc[3];
      efield_ubc[3] = efield_ubc[2];
      efield_ubc[2] = efield_ubc[1];
      efield_ubc[1] = efield_ubc[0];
      efield_ubc[0] = efield[xv_nopo_calc-1];

      ///////////////////////////////////////////////////////////////
      ///////////////// insert source into E field //////////////////
      ///////////////////////////////////////////////////////////////

      efield[xv_src_idx] = efield[xv_src_idx] + source_calc[jj];

    }

    now = std::chrono::high_resolution_clock::now();
    double timing_ms = std::chrono::duration<double, std::milli>(now-timing).count();
    double current_time_ms = std::chrono::duration<double, std::milli>(now-total_timing).count();
    double time_remaining = (current_time_ms * tv_nopo) / (ii+1) - current_time_ms;
    ms_format(time_remaining);
    timing = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for private(jj)
    for (jj = 0; jj < xv_nopo-1; jj++)
    {
      efield_mat_real[(ii+1)*xv_nopo + jj] = efield[jj*xv_sep].real();
      efield_mat_imag[(ii+1)*xv_nopo + jj] = efield[jj*xv_sep].imag();
      hfield_mat_real[(ii+1)*xv_nopo + jj] = (1/nu)*hfield[jj*xv_sep].real();
      hfield_mat_imag[(ii+1)*xv_nopo + jj] = (1/nu)*hfield[jj*xv_sep].imag();
    }

  }

  h5.hdf5_write_vector_double(filename, "efield_mat_real", efield_mat_real, tv_nopo*xv_nopo);
  h5.hdf5_write_vector_double(filename, "efield_mat_imag", efield_mat_imag, tv_nopo*xv_nopo);
  h5.hdf5_write_vector_double(filename, "hfield_mat_real", hfield_mat_real, tv_nopo*xv_nopo);
  h5.hdf5_write_vector_double(filename, "hfield_mat_imag", hfield_mat_imag, tv_nopo*xv_nopo);

}

double sellmeier_fused_silica(double l0)
{
    l0 = 1e6 * l0;
    double n_sqr = 1 + ((0.6961663*pow(l0,2)) / (pow(l0,2) - pow(0.0684043,2))) + ((0.4079426*pow(l0,2)) / (pow(l0,2) - pow(0.1162414,2))) + ((0.8974794*pow(l0,2)) / (pow(l0,2) - pow(9.896161,2)));
    return sqrt(n_sqr);
}

void ms_format(double ms)
{

  int time_target = ms / 1000;
  int hour = time_target / 3600;
  int second = time_target % 3600;
  int minute = second / 60;
  second = second % 60;

  int ms_sub = 1000 * (second + minute * 60 + hour * 3600);
  int ms_remain = ms - ms_sub;

  printf("Approximate time remaining: %.2d:%.2d:%.2d:%.6d\n",hour,minute,second,ms_remain);

}
