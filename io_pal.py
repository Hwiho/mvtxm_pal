# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:55:41 2021

@author: Hwiho Kim
"""
import numpy as np
from tqdm import trange
from scipy.signal import medfilt2d
from skimage import io
import multiprocessing as mp
from functools import partial
import os , h5py
import glob
from tqdm.contrib.concurrent import process_map


class iotools_pal():

    def __init__(self):

        self.img = None
        self.bkg = None
        self.eng = None
        self.fn = None
        self.bkg_sub = None
        self.shape = None
        self.medfilt = None
        self.proj_dir = None
        self.bkg_dir = None

    def load_processed_h5(self, fn = None, index = None, eng = None, wl_flag = None):
        if fn == None:
            self.fn = str(input("Insert the processed h5 filename (hdf5) : "))
        else:
            self.fn = fn
        if self.fn[-3:] == '.h5':
            with h5py.File(self.fn, 'r') as f:
                self.img = np.array(f['Img_aligned'])
                self.bkg_sub = np.array(f['Img_aligned'])
                if "Energy" in f:
                    self.eng = np.round(np.array(f["Energy"]))
                else:
                    print("Energy does not exist in the HDF5 file.")
                    if len(self.img) == 21:
                        self.eng = np.arange(8340, 8361)
                if index != None:
                    self.img = self.img[index[0]:index[1]]
                    self.eng = self.eng[index[0]:index[1]]
                print(self.eng)
                if wl_flag is not None:
                    self.fit = np.array(f['Peaksite'])
        else:
            self.img = io.imread(self.fn)[index[0]:index[1]]
            self.eng = eng
            print("Tiff loaded")
            if len(self.img) == 21:
                self.eng = np.arange(8340, 8361)
            print(self.eng)

    def select_directory(self):
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        proj_dir = filedialog.askdirectory(parent=root, initialdir=r"D:\New", title='Please select the projection directory')
        print("\nprojection directory : ", proj_dir)
        bkg_dir = filedialog.askdirectory(parent=root, initialdir=r"D:\New", title='Please select the background directory')
        print("background directory : ", bkg_dir)
        self.proj_dir = proj_dir
        self.bkg_dir = bkg_dir

    def img_avg(self, avg):
        proj_fn = sorted(glob.glob(self.proj_dir + '/*.tif'));
        bkg_fn = sorted(glob.glob(self.bkg_dir + '/*.tif'))
        self.shape = io.imread(proj_fn[0]).shape
        proj_image = np.zeros((len(proj_fn),self.shape[0],self.shape[1]),dtype = np.uint16)
        bkg_image = np.zeros((len(bkg_fn),self.shape[0],self.shape[1]),dtype = np.uint16)
        for k in trange(len(proj_fn)):
            proj_image[k] = io.imread(proj_fn[k])
        for l in trange(len(bkg_fn)):
            bkg_image[l] = io.imread(bkg_fn[l])
        for e in [self.proj_dir,self.bkg_dir]:
            if not os.path.exists(e + "/Average/"):
                os.makedirs(e + "/Average/")
        print("Image Averaging...")

        for i in trange(0, len(proj_fn), avg):
            avg_proj = np.mean(proj_image[i:i + avg], axis=0).astype(np.uint16)
            io.imsave(self.proj_dir + "/Average/" + "Average_" + proj_fn[i].split('\\')[-1], avg_proj)
        for i in trange(0, len(bkg_fn), avg):
            avg_bkg = np.mean(bkg_image[i:i + avg], axis=0).astype(np.uint16)
            io.imsave(self.bkg_dir + "/Average/" + "Average_" + bkg_fn[i].split('\\')[-1], avg_bkg)
        self.proj_dir = self.proj_dir + "/Average/"
        self.bkg_dir = self.bkg_dir + "/Average/"

    def fn_PAL(self):
        self.fn = os.path.basename(os.path.normpath(self.proj_dir))

    def load_tif(self):
        import glob
        proj_fn = sorted(glob.glob(self.proj_dir + '/*.tif'));
        bkg_fn = sorted(glob.glob(self.bkg_dir + '/*.tif'))
        self.shape = io.imread(proj_fn[0]).shape
        proj_image = np.zeros((len(proj_fn),self.shape[0],self.shape[1]))
        bkg_image = np.zeros((len(bkg_fn),self.shape[0],self.shape[1]))

        for i in trange(len(proj_fn)):
            proj_image[i] = io.imread(proj_fn[i])
            bkg_image[i] = io.imread(bkg_fn[i])

        self.img = proj_image
        self.bkg = bkg_image

    def load_stack_and_energy(self, dir_path):
        self.img = io.imread(os.path.join(dir_path, 'mean_proj_stack.tif')).astype(np.float32)
        self.bkg = io.imread(os.path.join(dir_path, 'mean_back_stack.tif')).astype(np.float32)
        self.eng = np.loadtxt(os.path.join(dir_path, 'energy.txt'))
        print(f"Loaded {len(self.img)} images, energy: {self.eng}")

    def energy_list_extraction(self):
        fn = glob.glob(self.proj_dir + '/*.tif')
        energy_sort_dict = {}
        energy = []
        for key in range(len(fn)):
            energy_sort_dict[str(key)] = fn[key].split('_')
            energy.append(float(energy_sort_dict[str(key)][energy_sort_dict[str(key)].index('eV') - 1]))
        self.eng = np.array(sorted(energy))

    def rmv_bkg(self):
        self.bkg_sub = -np.log10(self.img / self.bkg)
        print("Background Subtraction Done!")

    def median_filt(self, kernel, mp_flag=True):
        print("Median filtering...")
        self.shape = self.img.shape
        filt = np.zeros(self.shape)
        if kernel == 1:
            print('Kernel = (1x1) : pass')
            self.medfilt = self.bkg_sub
            pass
        else:
            if len(self.shape) == 2:
                medfilt = self.img.reshape(1, self.shape[0], self.shape[1])
            else:
                if mp_flag:
                    n_cpu = os.cpu_count() // 2
                    with mp.get_context('spawn').Pool(n_cpu) as pool:
                        filt = np.array(pool.map(partial(medfilt2d, kernel_size=kernel),
                                        [self.bkg_sub[i] for i in range(len(self.img))]))
                else:
                    for i in trange(self.shape[0]):
                        filt[i] = medfilt2d(self.bkg_sub[i], kernel)

            self.medfilt = filt


