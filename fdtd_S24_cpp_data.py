# -*- coding: utf-8 -*-
"""
@author: t.maybour
"""

import os
import h5py
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def main():

    plots = cplots()

    c = np.float64(299792458) # speed of light
    epsilon0 = np.float(8.85418782e-12)
    mu0 = np.float(1.25663706e-6)
    nu = np.sqrt(mu0 / epsilon0)

#########################################################################

    with h5py.File('fdtd_data.h5', "r") as hdf:

        ################### Inputs ###################
        bw_nopo = np.uint32(hdf.get('bw_nopo'))
        tv_nopo_input = np.uint32(hdf.get('tv_nopo_input'))
        xv_nopo_input = np.uint32(hdf.get('xv_nopo_input'))
        res = np.uint32(hdf.get('res'))
        ##############################################
        l0 = np.float64(hdf.get('l0'))
        pfbw = np.float64(hdf.get('pfbw'))
        ##############################################
        glen = np.float64(hdf.get('glen'))
        gstart = np.float64(hdf.get('gstart'))
        dn = np.float64(hdf.get('dn'))
        ds = np.float64(hdf.get('ds'))
        gramp_scale = np.float64(hdf.get('gramp_scale'))
        ##############################################
        tlen = np.float64(hdf.get('tlen'))
        mu = np.float64(hdf.get('mu'))
        tstart = np.float64(hdf.get('tstart'))
        delta_t = np.float64(hdf.get('delta_t'))
        ton = np.float64(hdf.get('ton'))
        ##############################################

        ################### DATA #####################
        tv_nopo = np.uint32(hdf.get('tv_nopo'))
        xv_nopo = np.uint32(hdf.get('xv_nopo'))
        n0 = np.float64(hdf.get('n0'))
        gfstart = np.float64(hdf.get('gfstart'))
        gfend = np.float64(hdf.get('gfend'))
        gend = np.float64(hdf.get('gend'))
        ##############################################
        kv = np.array(hdf.get('kv'), dtype=np.float64)
        fk = np.array(hdf.get('fk'), dtype=np.float64)
        xv = np.array(hdf.get('xv'), dtype=np.float64)
        apod = np.array(hdf.get('apod'), dtype=np.float64)
        gm = np.array(hdf.get('gm'), dtype=np.float64)
        tv = np.array(hdf.get('tv'), dtype=np.float64)
        twindow = np.array(hdf.get('twindow'), dtype=np.float64)
        source_real = np.array(hdf.get('source_real'), dtype=np.float64)
        source_imag = np.array(hdf.get('source_imag'), dtype=np.float64)
        efield_mat_real = np.array(hdf.get('efield_mat_real'), dtype=np.float64)
        efield_mat_imag = np.array(hdf.get('efield_mat_imag'), dtype=np.float64)
        hfield_mat_real = np.array(hdf.get('hfield_mat_real'), dtype=np.float64)
        hfield_mat_imag = np.array(hdf.get('hfield_mat_imag'), dtype=np.float64)
        ##############################################

#########################################################################

    efield_mat = np.zeros((tv_nopo,xv_nopo), dtype=np.complex128)
    hfield_mat = np.zeros((tv_nopo,xv_nopo), dtype=np.complex128)

    for ii in range(0, tv_nopo):
        for jj in range(0, xv_nopo):
            efield_mat[ii,jj] = efield_mat_real[ii*xv_nopo+jj] + 1j*efield_mat_imag[ii*xv_nopo+jj]
            hfield_mat[ii,jj] = hfield_mat_real[ii*xv_nopo+jj] + 1j*hfield_mat_imag[ii*xv_nopo+jj]

    edensity = np.zeros((tv_nopo, xv_nopo), dtype=np.float64)
    edensity[:,:] = np.real((n0**2)*np.conj(efield_mat[:,:])*(efield_mat[:,:]) + (c**2)*np.conj(hfield_mat[:,:])*(hfield_mat[:,:]))
    edensity[:,:] = edensity[:,:] / np.max(edensity[:,:])

    source = np.zeros(tv_nopo, dtype=np.complex128)
    source[:] = source_real[:] + 1j*source_imag[:]

#######################################################################
############################## Graphs #################################
#######################################################################

    plots.x_label = 'Wavenumber (1/m)'
    plots.line_plot(kv, fk/np.max(fk), filename='fk')

    plots.x_scale = 1e3
    plots.x_label = 'Length (mm)'

    plots.line_plot(xv, apod, filename='apod')
    plots.line_plot(xv, gm, filename='gm')

    plots.x_scale = 1e12
    plots.x_label = 'Time (ps)'

    plots.line_plot(tv, twindow, filename='twindow')
    plots.line_plot(tv, np.abs(source)**2/np.max(np.abs(source)**2), filename='source')

########################################################################

    plots.width = 32
    plots.height = 16
    plots.axes_label_size = 50
    plots.tick_label_size = 45
    plots.major_tick_length = 8
    plots.major_tick_width = 3

    plots.x_scale = 1e12
    plots.x_label = 'Time (ps)'
    plots.y_scale = 1e3
    plots.y_label = 'Position (mm)'

    plots.z_label = "Energy Density (a.u.)"

    plots.heat_map(tv, xv, np.transpose(edensity), filename='edensity_plot')

#################################################################################
#################################################################################
#################################################################################

class cplots():

    def __init__(self):
        self.filename = ''
        self.save_dir = ''
        self.width = 20
        self.height = 16
        self.axes_label_size = 35
        self.tick_label_size = 35
        self.axes_label_padding = 35
        self.margins = False
        self.x_label = ''
        self.y_label = ''
        self.z_label = ''
        self.x_scale = 1
        self.y_scale = 1
        self.z_scale = 1
        self.linewidth = 6
        self.major_tick_length = 5
        self.major_tick_width = 3
        self.tick_label_padding = 15

    def set_save_dir(self):
        if (len(self.save_dir) > 0) :
            if (self.save_dir[-1] != '/'):
                self.save_dir = self.save_dir + '/'
        else:
            self.save_dir = 'graphs/'
        self.create_dir(self.save_dir)

    def create_dir(self, dir_in):
        if (dir_in[-1] != '/'):
            dir_in = dir_in + '/'
        if not os.path.exists(dir_in):
            os.makedirs(dir_in)

    def axis_setup(self):
        plt.rc('axes', labelsize=self.axes_label_size)
        plt.rc('xtick', labelsize=self.tick_label_size)
        plt.rc('ytick', labelsize=self.tick_label_size)

    def set_kwargs(self, kwargs):
        self.filename = kwargs.get('filename', self.filename)
        self.save_dir = kwargs.get('save_dir', self.save_dir)

    def line_plot(self, x, data, *args, **kwargs):

        self.set_kwargs(kwargs)
        self.set_save_dir()
        self.axis_setup()

        y = np.array(data)

        plt.figure(figsize=(self.width, self.height))
        plt.ticklabel_format(useOffset=False)
        if self.margins == False:
            plt.margins(x=0)
        plt.xlabel(self.x_label, labelpad=self.axes_label_padding)
        plt.ylabel(self.y_label, labelpad=self.axes_label_padding)

        plt.plot(self.x_scale*x,self.y_scale*y,linewidth=self.linewidth)

        plt.tight_layout()
        plt.savefig(self.save_dir + self.filename + ".png")
        plt.close()

    def heat_map(self, x, y, z, *args, **kwargs):

        self.set_kwargs(kwargs)
        self.set_save_dir()
        self.axis_setup()

        u, v = np.meshgrid(x, y)

        fig, ax1 = plt.subplots(constrained_layout=True)

        ax1.tick_params(direction='out', length=self.major_tick_length, width=self.major_tick_width, labelsize=self.axes_label_size, pad=self.tick_label_padding)
        im = ax1.pcolormesh(self.x_scale*u, self.y_scale*v, self.z_scale*z, cmap='viridis')

        ax1.set_xlabel(self.x_label, labelpad=self.axes_label_padding)
        ax1.set_ylabel(self.y_label, labelpad=self.axes_label_padding)

        cbar = fig.colorbar(im, ax=ax1) #need a colorbar to show the intensity scale
        cbar.set_label(self.z_label, labelpad=self.axes_label_padding)
        cbar.ax.tick_params(labelsize=self.axes_label_size)

        #################################################
        w=self.width
        h=self.height
        ax=plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)
        #################################################

        plt.savefig(self.save_dir + self.filename + ".png")
        plt.close()

#################################################################################
#################################################################################
#################################################################################

if __name__ == '__main__':
    main()
