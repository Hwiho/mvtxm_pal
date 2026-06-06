import glob
from skimage import io
import os
import numpy as np

class particle_analysis():

    def __init__(self, dir_path, ref):
        self.dir = dir_path
        self.dict = {}
        self.ref = ref
        self.basename = None
        self.load_img_stack()
        self.calculate_save()

    def load_img_stack(self):
        self.flist = glob.glob(self.dir + '/*.tif')

        for fn in self.flist:
            basename = os.path.basename(fn)[:-4]
            img = io.imread(fn)
            self.dict[f'{basename}'] = img

    def calculate_save(self):
        if not os.path.exists(self.dir + '/test.txt'):

            with open(self.dir + '/test.txt', 'w') as File:

                for basename, img in self.dict.items():
                    img[img>self.ref[1]] = self.ref[1]
                    img[(img < self.ref[0]) & (img > 0)] = self.ref[0]
                    mean = np.mean(img[img!=0])
                    std = np.std(img[img!=0])
                    coord = basename.split('-')[-1]
                    File.write(coord+'\t'+str(mean)+'\t'+str(std)+'\n')

        else:
            print("File exists")

if __name__ == "__main__":
    dir_path = "C:/Users/hwiho/Documents/Research paper/HT_paper/FF-XANES/ASSBs_HT_3_re_proj/Average/Data_ASSBs_HT_3_re/Multi_crop"
    ref = [8364, 8371]
    p_ana = particle_analysis(dir_path, ref)

