import numpy as np
import cv2
import os
import h5py
import glob

class img_util():

    def __init__(self, h5file):
        self.fn = h5file
        self.img = None
        self.shape = None
        self.img_resize = None

    def load_h5(self):
            
         with h5py.File(self.fn, 'r') as f:
            self.img = np.array(f['img_xanes'])
            self.bkg = np.array(f['img_bkg'])
            self.eng = np.array(f['X_eng'])
            self.dark = np.array(f['img_dark'])
            self.mag = np.array(f['Magnification'])
            self.pixel = np.array(f['Pixel Size'])
            self.note = str(f['note'])
            self.scan_id = str(f['scan_id'])
            self.scan_time = np.array(f['scan_time'])
            self.uid = str(f['uid'])

    def rmv_bkg(self):
        self.bkg_sub = (self.img - self.dark) / (self.bkg-self.dark)
        print("Background Subtraction Finished!")

    def binning(self, bin, interp):
        
        '''
        :param dim: tuple, target size
        :param bin: binning size ex) 2, 4
        :param interp:
                cv2.INTER_NEAREST: OMIT
                cv2.INTER_LINEAR: (2x2) kernel
                cv2.INTER_CUBIC: (4x4) kernel
                cv2.INTER_LANCZOS4: (8x8) kernel
                cv2.INTER_AREA: Area information efficent when reducing the size.
        :return: self.img_resize = resized image

        '''
        self.bin = bin
        self.shape = self.img.shape
        print(f"\nBinning : {int(bin)}x{int(bin)}")

        self.img_resize = np.zeros((self.shape[0],int(self.shape[1]/bin), int(self.shape[2]/bin)))
        dim = (self.shape[2]//bin, self.shape[1]//bin)
        for i in range(self.shape[0]):
            self.img_resize[i] = cv2.resize(self.bkg_sub[i], dsize=dim, interpolation=interp)

        print("Binning finished")

    def create_h5(self):
        
        dir = os.path.dirname(self.fn)
        if not os.path.exists(dir + f"/bin{self.bin}/"):
            os.makedirs(dir + f"/bin{self.bin}/")
            self.dir = dir + f"/bin{self.bin}/"
        else:
            self.dir = dir + f"/bin{self.bin}/"
        new_fn = os.path.basename(self.fn)
        with h5py.File(self.dir + new_fn[:-3] + f'_bin{self.bin}' + '.h5', 'w') as hf:
            hf.create_dataset('Magnification', data=self.mag)
            hf.create_dataset('Pixel Size', data=self.pixel)
            hf.create_dataset('X_eng', data=self.eng)
            hf.create_dataset('img_xanes', data=self.img_resize)
            hf.create_dataset('note', data=self.note)
            hf.create_dataset('scan_id', data=self.scan_id)
            hf.create_dataset('scan_time', data=self.scan_time)
            hf.create_dataset('uid', data=self.uid)
        print('h5file has been written')
def main():
    dir = '/data/hkim_data/LIBs_BNL_622_Insitu'
    h5list = sorted(glob.glob(dir + '/*.h5'))
    for h5file in h5list:
        util = img_util(h5file)
        util.load_h5()
        util.rmv_bkg()
        util.binning(bin = 4, interp=cv2.INTER_LANCZOS4)
        util.create_h5()
        name = os.path.basename(h5file)
        print(f'{name} finished')
if __name__ == '__main__':
    main()