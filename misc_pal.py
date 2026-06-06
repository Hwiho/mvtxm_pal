import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange
from skimage import io , color, img_as_ubyte
import os
import copy
import h5py
import glob

class data_mng_pal():

    def __init__(self, fitting, data_dir, fn):
        self.relrgb_uint8 = None
        self.rgb_uint8 = None
        self.rgb = fitting.rgb
        self.relrgb = fitting.relrgb
        self.fn = fn
        self.data_dir = data_dir
        self.scan_id = None
        self.scan_pos = None
        self.img_aligned = fitting.img
        self.thickness = fitting.thickness
        self.peaksite = fitting.peaksite
        self.mean = fitting.mean
        self.stdev = fitting.stdev
        self.eng = fitting.eng
        self.imgdir = None
        self.hisdir = None
        self.h5dir = None
        self.relimgdir = None
        self.fitting = fitting

    def save_id(self, fn):
        if self.fn[-3:] == '.h5':
            split_index = self.fn.split('_')
            self.scan_id = split_index[split_index.index('id') + 1]
            self.scan_pos = split_index[split_index.index('pos') + 1][:-3]
            self.data_dir = os.path.dirname(self.fn[:-3])
        elif self.fn[-5:] == '.tiff':
            split_index = self.fn.split('_')
            self.scan_id = split_index[split_index.index('sample') + 1]
            self.scan_pos = split_index[split_index.index('pos') + 1]
            self.data_dir = os.path.dirname(self.fn[:-3])
        elif self.fn[-4:] == '.tif':
            split_index = self.fn.split('_')
            self.scan_id = split_index[split_index.index('ml') + 1]
            self.scan_pos = split_index[split_index.index('pos') + 1]
            self.data_dir = os.path.dirname(self.fn[:-3])

    def create_dir_all_in_one(self, polynomial, fit_num):

        try:
            if polynomial == 2:
                self.all_dir = self.data_dir + f"/Data_folder_poly{polynomial}_fit{fit_num}"
            elif polynomial == None:
                self.all_dir = self.data_dir + f"/Data_folder_LCfit"
            else:
                self.all_dir = self.data_dir + f"/Data_foler_wl_poly{polynomial}"
            if not os.path.exists(self.all_dir):
                os.makedirs(self.all_dir + "/images")
                os.makedirs(self.all_dir + "/rel_images")
                os.makedirs(self.all_dir + "/h5files")
                os.makedirs(self.all_dir + "/histogram")

        except OSError:
            print("Error: Failed to create the directory.")

    def save_into_separate_folders_all_in_one(self, bins, range):

        self.imgdir = self.all_dir + "/images" + "/"
        self.relimgdir = self.all_dir + "/rel_images" + "/"
        self.hisdir = self.all_dir + "/histogram" + "/"
        self.h5dir = self.all_dir + "/h5files" + "/"
        self.rgb[self.rgb>1] = 1 ; self.relrgb[self.relrgb>1] =1;
        self.rgb[self.rgb<-1] = -1 ; self.relrgb[self.relrgb<-1] = -1;

        self.rgb_uint8 = img_as_ubyte(self.rgb)
        self.relrgb_uint8 = img_as_ubyte(self.relrgb)
        io.imsave(self.imgdir + "/colormap_id_" + self.scan_id + "_pos_" + self.scan_pos + ".png", self.rgb_uint8)
        io.imsave(self.relimgdir + "/relcolormap_id_" + self.scan_id + "_pos_" + self.scan_pos + ".png",
                  self.relrgb_uint8)

        self.hist, self.bins = np.histogram(self.peaksite, bins, range, weights=None)
        width = 0.7 * (self.bins[1] - self.bins[0])
        center = (self.bins[:-1] + self.bins[1:]) / 2
        fig, ax = plt.subplots()
        ax.bar(center, self.hist, align='center', width=width)
        fig.savefig(self.hisdir + 'histogram_id_' + self.scan_id + '_pos_' + self.scan_pos + '.png')

        with h5py.File(self.h5dir + "image_dataset_id_" + self.scan_id + '_pos_' + self.scan_pos + '.h5', 'w') as hf:
            hf.create_dataset("Img_aligned", data=self.img_aligned)
            hf.create_dataset("Peaksite", data=self.peaksite)
            hf.create_dataset("Thickness", data=self.thickness)
            hf.create_dataset("RGB_image", data=self.rgb)
            hf.create_dataset("Energy", data=self.eng)
            hf.create_dataset("Mean", data= self.mean)
            hf.create_dataset("STD", data = self.stdev)
            his = hf.create_group("Histogram")

    def save_id_PAL(self, fn):
        if self.fn[-3:] == '.h5':
            start = fn.find('id_') + len('id_')
            end = fn.find('MULTI')
            extracted_str = fn[start:end]
            self.scan_id = extracted_str
        else:
            self.scan_id = self.fn
        if 'MULTI' in self.fn:
            idx = self.fn.find('MULTI') + len('MULTI')
            num_str = ''
            while idx < len(fn) and fn[idx].isdigit():
                num_str += fn[idx]
                idx += 1
            self.scan_pos = num_str
        else:
            self.scan_pos = '00'

    def create_dir(self):
        try:
            if not os.path.exists(self.data_dir + "/Data_" + self.scan_id):
                os.makedirs(self.data_dir + "/Data_" + self.scan_id)
            self.data_dir = self.data_dir + "/Data_" + self.scan_id + '/'
        except OSError:
            print("Error: Failed to create the directory.")

    def save_color(self):
        self.rgb_uint8 = img_as_ubyte(self.rgb)
        self.relrgb_uint8 = img_as_ubyte(self.relrgb)
        io.imsave(self.data_dir + "colormap_id_" + self.scan_id + "_pos_" + self.scan_pos + ".png",self.rgb_uint8)
        io.imsave(self.data_dir + "relcolormap_id_" + self.scan_id + "_pos_" + self.scan_pos + ".png",
                  self.relrgb_uint8)
        io.imsave(self.data_dir + "gray_id_" + self.scan_id + "_pos_" + self.scan_pos + ".tif", self.thickness)
        if self.fitting.norm_img is not None:
            io.imsave(self.data_dir + "norm_stack_id_" + self.scan_id + "_pos_" + self.scan_pos + ".tif", self.fitting.norm_img)
            io.imsave(self.data_dir + "smth_stack_id_" + self.scan_id + "_pos_" + self.scan_pos + ".tif",
                      self.fitting.smth_img)

    def histogram (self, bins, range):
        self.hist, self.bins = np.histogram(self.peaksite, bins, range, weights=None)
        width = 0.7 * (self.bins[1] - self.bins[0])
        center = (self.bins[:-1] + self.bins[1:]) / 2
        fig, ax = plt.subplots()
        ax.bar(center, self.hist, align='center', width=width)
        #plt.show()
        fig.savefig(self.data_dir + 'Histogram_id_'+self.scan_id+'_pos_'+ self.scan_pos +'.png')

    def weighted_histogram(self, bins, range):
        weights = self.thickness / (np.max(self.thickness)-np.min(self.thickness))
        self.whist, self.wbins = np.histogram(self.peaksite, bins, range, weights=weights)
        width = 0.7 * (self.wbins[1] - self.wbins[0])
        center = (self.wbins[:-1] + self.wbins[1:]) / 2
        fig, ax = plt.subplots()
        ax.bar(center, self.whist, align='center', width=width)
        # plt.show()
        fig.savefig(self.data_dir + 'weighted_histogram_id_' + self.scan_id + '_pos_' + self.scan_pos + '.png')

    def create_h5(self):
        with h5py.File(self.data_dir + "Image_dataset_id_" + self.scan_id + '_pos_' + self.scan_pos + '.h5', 'w') as hf:
            hf.create_dataset("Img_aligned", data=self.img_aligned)
            hf.create_dataset("Peaksite", data=self.peaksite)
            hf.create_dataset("Thickness", data=self.thickness)
            hf.create_dataset("RGB_image", data=self.rgb)
            his = hf.create_group("Histogram")
            his.create_dataset("hist", data=self.hist)
            his.create_dataset("bins", data=self.bins)
            whis = hf.create_group("Weighted_histogram")
            whis.create_dataset("weighted_hist", data=self.whist)
            whis.create_dataset("weighted_bins", data=self.wbins)
            hf.create_dataset("Energy", data=self.eng)
            hf.create_dataset("Mean", data= self.mean)
            hf.create_dataset("STD", data = self.stdev)


class data_mng():

    def __init__(self, fitting, fn):
        self.relrgb_uint8 = None
        self.rgb_uint8 = None
        self.rgb = fitting.rgb
        self.relrgb = fitting.relrgb
        self.fn = fn
        self.data_dir = None
        self.scan_id = None
        self.scan_pos = None
        self.img_aligned = fitting.img
        self.thickness = fitting.thickness
        self.peaksite = fitting.peaksite
        self.mean = fitting.mean
        self.stdev = fitting.stdev
        self.eng = fitting.eng
        self.imgdir = None
        self.hisdir = None
        self.h5dir = None
        self.relimgdir = None

    def save_id(self, fn):
        if self.fn[-3:] == '.h5':
            split_index = self.fn.split('_')
            self.scan_id = split_index[split_index.index('id') + 1]
            self.scan_pos = split_index[split_index.index('pos') + 1][:-3]
            self.data_dir = os.path.dirname(self.fn[:-3])
        elif self.fn[-5:] == '.tiff':
            split_index = self.fn.split('_')
            self.scan_id = split_index[split_index.index('sample') + 1]
            self.scan_pos = split_index[split_index.index('pos') + 1]
            self.data_dir = os.path.dirname(self.fn[:-3])
        elif self.fn[-4:] == '.tif':
            split_index = self.fn.split('_')
            self.scan_id = split_index[split_index.index('ml') + 1]
            self.scan_pos = split_index[split_index.index('pos') + 1]
            self.data_dir = os.path.dirname(self.fn[:-3])

    def create_dir_all_in_one(self, polynomial, fit_num):

        try:
            if polynomial == 2:
                self.all_dir = self.data_dir + f"/Data_folder_poly{polynomial}_fit{fit_num}"
            elif polynomial == None:
                self.all_dir = self.data_dir + f"/Data_folder_LCfit"
            else:
                self.all_dir = self.data_dir + f"/Data_folder_wl_poly{polynomial}"
            if not os.path.exists(self.all_dir):
                os.makedirs(self.all_dir + "/images")
                os.makedirs(self.all_dir + "/rel_images")
                os.makedirs(self.all_dir + "/h5files")
                os.makedirs(self.all_dir + "/histogram")

        except OSError:
            print("Error: Failed to create the directory.")

    def create_dir(self):
        split_index = self.fn.split('_')
        if 'dataset' in split_index:
            if not os.path.exists(self.data_dir + "/Processed_" + self.scan_id + '_' + self.scan_pos + '/'):
                os.makedirs(self.data_dir + "/Processed_"  + self.scan_id + '_' + self.scan_pos + '/')
            self.data_dir = self.data_dir + "/Processed_" + self.scan_id + '_' + self.scan_pos + '/'
        elif self.fn[-5:] == '.tiff':
            if not os.path.exists(self.data_dir + "/Tiff_" + self.scan_id + '_' + self.scan_pos + '/'):
                os.makedirs(self.data_dir + "/Tiff_"  + self.scan_id + '_' + self.scan_pos + '/')
            self.data_dir = self.data_dir + "/Tiff_" + self.scan_id + '_' + self.scan_pos + '/'
        elif self.fn[-4:] == 'tif':
            if not os.path.exists(self.data_dir + "/Tif_" + self.scan_id + '_' + self.scan_pos + '/'):
                os.makedirs(self.data_dir + "/Tif_"  + self.scan_id + '_' + self.scan_pos + '/')
            self.data_dir = self.data_dir + "/Tif_" + self.scan_id + '_' + self.scan_pos + '/'
        else:
            try:
                if not os.path.exists(self.data_dir + "/Data_" + self.scan_id + '_' + self.scan_pos + '/'):
                    os.makedirs(self.data_dir + "/Data_" + self.scan_id + '_' + self.scan_pos + '/')
                self.data_dir = self.data_dir + "/Data_" + self.scan_id + '_' + self.scan_pos + '/'
            except OSError:
                print("Error: Failed to create the directory.")

    def save_into_separate_folders_all_in_one(self, bins, range):

        self.imgdir = self.all_dir + "/images" + "/"
        self.relimgdir = self.all_dir + "/rel_images" + "/"
        self.hisdir = self.all_dir + "/histogram" + "/"
        self.h5dir = self.all_dir + "/h5files" + "/"
        self.rgb[self.rgb>1] = 1 ; self.relrgb[self.relrgb>1] =1;
        self.rgb[self.rgb<-1] = -1 ; self.relrgb[self.relrgb<-1] = -1;

        self.rgb_uint8 = img_as_ubyte(self.rgb)
        self.relrgb_uint8 = img_as_ubyte(self.relrgb)
        io.imsave(self.imgdir + "/colormap_id_" + self.scan_id + "_pos_" + self.scan_pos + ".png", self.rgb_uint8)
        io.imsave(self.relimgdir + "/relcolormap_id_" + self.scan_id + "_pos_" + self.scan_pos + ".png",
                  self.relrgb_uint8)

        self.hist, self.bins = np.histogram(self.peaksite, bins, range, weights=None)
        width = 0.7 * (self.bins[1] - self.bins[0])
        center = (self.bins[:-1] + self.bins[1:]) / 2
        fig, ax = plt.subplots()
        ax.bar(center, self.hist, align='center', width=width)
        fig.savefig(self.hisdir + 'histogram_id_' + self.scan_id + '_pos_' + self.scan_pos + '.png')

        with h5py.File(self.h5dir + "image_dataset_id_" + self.scan_id + '_pos_' + self.scan_pos + '.h5', 'w') as hf:
            hf.create_dataset("Img_aligned", data=self.img_aligned)
            hf.create_dataset("Peaksite", data=self.peaksite)
            hf.create_dataset("Thickness", data=self.thickness)
            hf.create_dataset("RGB_image", data=self.rgb)
            his = hf.create_group("Histogram")

            his.create_dataset("hist", data=self.hist)
            his.create_dataset("bins", data=self.bins)
            hf.create_dataset("Energy", data=self.eng)

    def save_color(self):
        self.rgb_uint8 = img_as_ubyte(self.rgb)
        self.relrgb_uint8 = img_as_ubyte(self.relrgb)
        self.thick_uint8 = img_as_ubyte(self.thickness)
        io.imsave(self.data_dir + "colormap_id_" + self.scan_id + "_pos_" + self.scan_pos + ".png", self.rgb_uint8)
        io.imsave(self.data_dir + "relcolormap_id_" + self.scan_id + "_pos_" + self.scan_pos + ".png",
                  self.relrgb_uint8)
        io.imsave(self.data_dir + "gray_id_" + self.scan_id + "_pos_" + self.scan_pos + ".png", self.thick_uint8)

    def histogram(self, bins, range):
        self.hist, self.bins = np.histogram(self.peaksite, bins, range, weights=None)
        width = 0.7 * (self.bins[1] - self.bins[0])
        center = (self.bins[:-1] + self.bins[1:]) / 2
        fig, ax = plt.subplots()
        ax.bar(center, self.hist, align='center', width=width)
        # plt.show()
        fig.savefig(self.data_dir + 'histogram_id_' + self.scan_id + '_pos_' + self.scan_pos + '.png')

    def weighted_histogram(self, bins, range):
        weights = self.thickness / (np.max(self.thickness)-np.min(self.thickness))
        self.whist, self.wbins = np.histogram(self.peaksite, bins, range, weights=weights)
        width = 0.7 * (self.wbins[1] - self.wbins[0])
        center = (self.wbins[:-1] + self.wbins[1:]) / 2
        fig, ax = plt.subplots()
        ax.bar(center, self.whist, align='center', width=width)
        # plt.show()
        fig.savefig(self.data_dir + 'weighted_histogram_id_' + self.scan_id + '_pos_' + self.scan_pos + '.png')

    def create_h5(self):
        with h5py.File(self.data_dir + "Image_dataset_id_" + self.scan_id + '_pos_' + self.scan_pos + '.h5', 'w') as hf:
            hf.create_dataset("Img_aligned", data=self.img_aligned)
            hf.create_dataset("Peaksite", data=self.peaksite)
            hf.create_dataset("Thickness", data=self.thickness)
            hf.create_dataset("RGB_image", data=self.rgb)
            his = hf.create_group("Histogram")
            his.create_dataset("hist", data=self.hist)
            his.create_dataset("bins", data=self.bins)
            whis = hf.create_group("Weighted_histogram")
            whis.create_dataset("weighted_hist", data=self.whist)
            whis.create_dataset("weighted_bins", data=self.wbins)
            hf.create_dataset("Energy", data=self.eng)
            hf.create_dataset("Mean", data= self.mean)
            hf.create_dataset("STD", data = self.stdev)

    def create_txt(self):
        File = open(self.data_dir + "/mean_std.txt", "a+")
        File.write(f'Scan_id : {self.scan_id}_{self.scan_pos}' +
                   f'\nMean : {np.mean(self.peaksite[self.peaksite != 0])}' +
                   f"\nStd : {self.stdev}")
        File.close()
