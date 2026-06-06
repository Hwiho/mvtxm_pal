import multiprocessing as mp
import os
import gc
from txm_sandbox.utils import xanes_regtools as xr
from txm_sandbox.utils.reg_algs import mrtv_mpc_combo_reg, mrtv_reg, mrtv_ls_combo_reg, shift_img
import numpy as np
from scipy.ndimage import fourier_shift
from functools import partial
from pystackreg import StackReg
import tifffile, h5py
from skimage.registration import phase_cross_correlation
from silx.io.dictdump import dicttoh5, h5todict
from tqdm.contrib.concurrent import process_map
import math

class regist ():

    def __init__(self, img, dtype='2D_XANES', **kwargs):
        self.chunk_sz = 2
        self.data_type = dtype.upper()
        self.eng_list = None
        self.eng_dict = {}
        self.alg_mrtv_level = 5
        self.alg_mrtv_width  = 10
        self.alg_mrtv_sp_kernel = 3
        self.alg_mrtv_sp_wz = 8
        self.anchor = 0
        self.chunks = {}
        self.img_ids = None
        self.img_ids_dict = {}
        self.mask = None
        self.use_mask = False
        self.overlap_ratio = 0.3
        self.fixed = None
        self.img = img
        self.N_CPU = os.cpu_count()//2
        self.shift_dict = {}
        self.shift = None
        self.abs_shifted_image = np.zeros(self.img.shape)

    def set_indices(self, img_id_s, img_id_e, fixed_img_id):
        self.img_id_s = img_id_s
        self.img_id_e = img_id_e
        self.fixed_img_id = fixed_img_id

    def set_ref_mode(self, ref_mode):
        self.ref_mode = ref_mode.upper()

    def set_eng (self, eng):
        self.eng_list = eng

    def set_method(self, method):
        self.method = method.upper()

    def set_reg_options(self, use_mask=True, mask_thres=0,
                        use_chunk=True, chunk_sz=5,
                        use_smooth_img=False, smooth_sigma=0,
                        mrtv_level=5, mrtv_width=8,
                        mrtv_sp_wz=8, mrtv_sp_kernel=3):
        """
        the current use_anchor setting is not contraversial. all the registrations
        use some anchors. It is not anchor but chunk setting making differences.
        The code shoudl be modified to have use_chunk and remove use_anchor

        Parameters
        ----------
        use_mask : TYPE, optional
            DESCRIPTION. The default is True.
        mask_thres : TYPE, optional
            DESCRIPTION. The default is 0.
        use_chunk : TYPE, optional
            DESCRIPTION. The default is True.
        anchor_id : TYPE, optional
            DESCRIPTION. The default is 0.
        use_smooth_img : TYPE, optional
            DESCRIPTION. The default is False.
        smooth_sigma : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        self.use_mask = use_mask
        self.mask_thres = mask_thres
        self.use_chunk = use_chunk
        self.chunk_sz = chunk_sz
        self.use_smooth_img = use_smooth_img
        self.img_smooth_sigma = smooth_sigma
        self.alg_mrtv_level = mrtv_level
        self.alg_mrtv_width  = mrtv_width
        self.alg_mrtv_sp_wz = mrtv_sp_wz
        self.alg_mrtv_sp_kernel = mrtv_sp_kernel


    def compose_dicts(self):
        if self.data_type == '3D_XANES':
            if self.fixed_img_id in range(self.img_id_s, self.img_id_e):
                self.anchor = self.fixed_img_id-self.img_id_s
                self.data_pnts = self.img_id_e - self.img_id_s
                print(self.img_id_s, self.img_id_e)
                print(len(self.img_ids))
                print(self.img_ids)
                print(self.fixed_img_id-self.img_id_s)
                self.anchor_id = self.img_ids[self.fixed_img_id-self.img_id_s]
                cnt = 0
                for ii in self.img_ids:
                    self.eng_dict[str(cnt).zfill(3)] = self.eng_list[cnt]
                    self.img_ids_dict[str(cnt).zfill(3)] = ii
                    cnt += 1
            else:
                print('fixed_img_id is outside of [img_id_s, img_id_e].')
        elif self.data_type == '2D_XANES':
            if self.fixed_img_id in range(self.img_id_s, self.img_id_e):
                self.anchor = self.fixed_img_id - self.img_id_s
                self.data_pnts = self.img_id_e - self.img_id_s
                self.img_ids = np.arange(self.img_id_s, self.img_id_e)
                self.anchor_id = self.fixed_img_id
                cnt = 0
                for ii in range(self.img_id_s, self.img_id_e):
                    self.eng_dict[str(cnt).zfill(3)] = self.eng_list[cnt]
                    self.img_ids_dict[str(cnt).zfill(3)] = ii
                    cnt += 1
            else:
                print('fixed_img_id is outside of [img_id_s, img_id_e].')

    def _chunking(self):
        """
        self.data_pnts: relative number defined as self.img_id_e - self.img_id_s + 1
        self.anchor: relative number defined as self.fixed_img_id - self.img_id_s

        self.chunks: generated variable; starting and ending idx of each chunk.
                     the idx are relative idx from self.img_id_s
        Returns
        -------
        None.

        """
        if self.use_chunk:
            right_num_chunk = int(np.ceil((self.data_pnts -
                                           self.anchor) / self.chunk_sz))
            left_num_chunk = int(np.ceil(self.anchor / self.chunk_sz))
            num_chunk = left_num_chunk + right_num_chunk
            self.num_chunk = num_chunk
            self.left_num_chunk = left_num_chunk - 1
            self.anchor_chunk = left_num_chunk - 1
            for ii in range(left_num_chunk-1):
                self.chunks[left_num_chunk-1-ii] = {'chunk_s':\
                    self.anchor - self.chunk_sz * (ii + 1) + 1}
                self.chunks[left_num_chunk-1-ii]['chunk_e'] =\
                    self.anchor - self.chunk_sz * ii

            if (self.anchor % self.chunk_sz) != 0:
                self.chunks[0] = {'chunk_s': 0}
                self.chunks[0]['chunk_e'] = int(self.anchor %
                                                self.chunk_sz)
            else:
                self.chunks[0] = {'chunk_s': 0}
                self.chunks[0]['chunk_e'] = self.chunk_sz

            for ii in range(left_num_chunk, num_chunk-1):
                self.chunks[ii] = {'chunk_s': self.anchor +\
                    self.chunk_sz * (ii - left_num_chunk) + 1}
                self.chunks[ii]['chunk_e'] = self.anchor +\
                    self.chunk_sz * (ii - left_num_chunk + 1)

            if ((self.data_pnts - self.anchor) % self.chunk_sz) != 1:
                self.chunks[num_chunk-1] = {'chunk_s':\
                    self.chunks[num_chunk-2]['chunk_e'] + 1}
                self.chunks[num_chunk-1]['chunk_e'] =\
                    self.data_pnts - 1
            else:
                self.chunks[num_chunk-1] = {'chunk_s':\
                    self.data_pnts - 1}
                self.chunks[num_chunk-1]['chunk_e'] =\
                    self.data_pnts - 1
        else:
            self.chunks[0] = {'chunk_s': 0}
            self.chunks[0]['chunk_e'] = self.data_pnts - 1

    def _alignment_scheduler(self, dtype='2D_XANES'):
        """
        2D XANES: [chunk_sz, img_s,     img_e,     ref_mode, fixed_img]
        3D XANES: [chunk_sz, scan_id_s, scan_id_e, ref_mode, fixed_scan_id]

        According to chunk_sz and ref_mode, we should make a list of pairs for
        comparison. imgs/scan_id_s and img_e/scan_id_e are used to determine
        the bounds to the numbers in the pairs. fixed_img/fixed_scan_id are the
        image/scan used as the anchor in the aligned image/recon sequence.
        These two numbers also affect the list fabrication.

        ref_mode: 'single', 'neighbor', 'average'
            'single': the last images in two consecutive chunks will be
                      compared and aligned. The fixed_img/fixed_scan_id are
                      anchored as the last image in its chunk. The chunks are
                      propagated to the right and left. So, the list of pairs
                      should look like
                      [[fixed_img, left_neighbor_chunk_last_img],
                       [left_neighbor_chunk_last_img, its_left_neighbor],
                       ...,
                       [fixed_img, right_neighbor_chunk_last_img],
                       [right_neighbor_chunk_last_img, its_right_neighbor],
                       ...,
                       [each_pair_in_each_chunk_with_last_img]
                      ]
            'neighbor': the neighbor images in two consecutive chunks will be
                      compared and aligned. The fixed_img/fixed_scan_id are
                      anchored as the last image in its chunk. The chunks are
                      propagated to the right and left. So, the list of pairs
                      should look like
                      [[fixed_img, first_img_in_same_chunk],
                       [first_img_in_same_chunk, its_left_neighbor],
                       ...,
                       [fixed_img, first_img_in_right_neighbor],
                       [first_img_in_right_neighbor, last_img_in_its_chunk],
                       ...,
                       [each_pair_in_each_chunk_with_last_img]
                      ]
        self.alignment_pair_list: generated variable with this function. It defines
                                  pairs of imgs for shift calculation. the idx of
                                  each pair are relative to self.img_id_s
        """
        self._chunking()
        self.alignment_pair_list = []

        if self.use_chunk:
            if self.ref_mode.upper() == 'SINGLE':
                # inter-chunk alignment pair
                for ii in range(self.left_num_chunk):
                    self.alignment_pair_list.append([self.chunks[self.left_num_chunk-ii]['chunk_e'],
                                                self.chunks[self.left_num_chunk-ii-1]['chunk_e']])
                self.alignment_pair_list.append([self.anchor_chunk,
                                            self.anchor_chunk])
                print(self.left_num_chunk, self.num_chunk)
                for ii in range(self.left_num_chunk+1, self.num_chunk):
                    self.alignment_pair_list.append([self.chunks[ii-1]['chunk_e'],
                                                self.chunks[ii]['chunk_e']])
                # intra-chunk alignment pair
                for ii in range(self.num_chunk):
                    for jj in range(self.chunks[ii]['chunk_s'],
                                    self.chunks[ii]['chunk_e']+1):
                        self.alignment_pair_list.append([self.chunks[ii]['chunk_e'], jj])

                tem = []
                for ii in self.alignment_pair_list:
                    if ii[0] == ii[1]:
                        tem.append(ii)
                for ii in tem:
                    self.alignment_pair_list.remove(ii)
                self.alignment_pair_list.append([self.anchor, self.anchor])
            elif self.ref_mode.upper() == 'NEIGHBOR':
                # inter-chunk alignment pair
                for ii in range(self.left_num_chunk):
                    self.alignment_pair_list.append([self.chunks[self.left_num_chunk-ii]['chunk_e'],
                                                self.chunks[self.left_num_chunk-ii]['chunk_s']])
                    self.alignment_pair_list.append([self.chunks[self.left_num_chunk-ii]['chunk_s'],
                                                self.chunks[self.left_num_chunk-ii-1]['chunk_e']])
                self.alignment_pair_list.append([self.chunks[self.anchor_chunk]['chunk_e'],
                                            self.chunks[self.anchor_chunk]['chunk_e']+1])
                for ii in range(self.left_num_chunk+1, self.num_chunk-1):
                    self.alignment_pair_list.append([self.chunks[ii]['chunk_s'],
                                                self.chunks[ii]['chunk_e']])
                    self.alignment_pair_list.append([self.chunks[ii]['chunk_e'],
                                                self.chunks[ii+1]['chunk_s']])
                self.alignment_pair_list.append([self.chunks[self.num_chunk-1]['chunk_s'],
                                            self.chunks[self.num_chunk-1]['chunk_e']])
                # inter-chunk alignment pair
                for ii in range(self.num_chunk):
                    for jj in range(self.chunks[ii]['chunk_s'],
                                    self.chunks[ii]['chunk_e']+1):
                        self.alignment_pair_list.append([self.chunks[ii]['chunk_e'], jj])

                tem = []
                for ii in self.alignment_pair_list:
                    if ii[0] == ii[1]:
                        tem.append(ii)
                for ii in tem:
                    self.alignment_pair_list.remove(ii)
                self.alignment_pair_list.append([self.anchor, self.anchor])
        else:
            for ii in range(self.data_pnts-1):
                self.alignment_pair_list.append([ii, ii+1])
            self.alignment_pair_list.append([self.anchor, self.anchor])

    def set_shift(self):
        self.shift = np.ndarray((len(self.alignment_pair_list), 2))

    def set_shift_dict(self):
        for key, item in self.img_ids_dict.items():
            self.shift_dict[str(item)] = self.shift[int(key)]

    def _sort_absolute_shift(self, shift_dict=None, optional_shift_dict=None):
        """
        self.shift_chain_dict: generated variables with this function. the idx
                               each img is associated with a chain of other img
                               idx with which the img shift is uniquely defined
                               relative to the anchor img. all idx are relative
                               to self.img_id_s

        Parameters
        ----------
        trialfn : TYPE
            DESCRIPTION.
        shift_dict : TYPE, optional
            DESCRIPTION. The default is None.
        optional_shift_dict : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if self.data_type.upper() == "3D_XANES":
            self.shift_chain_dict = {}
            for ii in range(len(self.alignment_pair_list)-1, -1, -1):
                self.shift_chain_dict[self.alignment_pair_list[ii][1]] = [self.alignment_pair_list[ii][0]]
                jj = ii - 1
                while jj>=0:
                    if self.shift_chain_dict[self.alignment_pair_list[ii][1]][-1] == self.alignment_pair_list[jj][1]:
                        self.shift_chain_dict[self.alignment_pair_list[ii][1]].append(self.alignment_pair_list[jj][0])
                    jj -= 1
            abs_shift_dict = {}
            if self.method.upper() == "SR":
                for key, item in self.shift_chain_dict.items():
                    item.insert(0, key)
                    shift = np.identity(3)
                    slioff = 0
                    for ii in range(len(item)-1):
                        idx = self.alignment_pair_list.index([item[ii+1], item[ii]])
                        shift = np.matmul(shift, np.array(shift_dict[str(idx)][1]))
                        slioff += int(optional_shift_dict[str(idx)][0])
                    abs_shift_dict[str(key).zfill(3)] = {'in_sli_shift':shift, 'out_sli_shift':slioff}
            else:
                for key, item in self.shift_chain_dict.items():
                    item.insert(0, key)
                    shift = 0.
                    slioff = 0
                    for ii in range(len(item)-1):
                        idx = self.alignment_pair_list.index([item[ii+1], item[ii]])
                        shift += np.array(shift_dict[str(idx)][1:])
                        slioff += int(shift_dict[str(idx)][0])
                    abs_shift_dict[str(key).zfill(3)] = {'in_sli_shift':shift, 'out_sli_shift':slioff}

        elif self.data_type.upper() == "2D_XANES":
            self.shift_chain_dict = {}
            for ii in range(len(self.alignment_pair_list)-1, -1, -1):
                self.shift_chain_dict[self.alignment_pair_list[ii][1]] = [self.alignment_pair_list[ii][0]]
                jj = ii - 1
                while jj>=0:
                    if self.shift_chain_dict[self.alignment_pair_list[ii][1]][-1] == self.alignment_pair_list[jj][1]:
                        self.shift_chain_dict[self.alignment_pair_list[ii][1]].append(self.alignment_pair_list[jj][0])
                    jj -= 1
            abs_shift_dict = {}
            if self.method.upper() == "SR":
                for key, item in self.shift_chain_dict.items():
                    item.insert(0, key)
                    shift = np.identity(3)
                    for ii in range(len(item)-1):
                        idx = self.alignment_pair_list.index([item[ii+1], item[ii]])
                        shift = np.matmul(shift, np.array(shift_dict[str(idx)]))
                    abs_shift_dict[str(key).zfill(3)] = {'in_sli_shift':shift}
            else:
                for key, item in self.shift_chain_dict.items():
                    item.insert(0, key)
                    print(key, item)
                    shift = 0.
                    for ii in range(len(item)-1):
                        idx = self.alignment_pair_list.index([item[ii+1], item[ii]])
                        shift += np.array(self.shift_dict[str(idx)])
                    abs_shift_dict[str(key).zfill(3)] = {'in_sli_shift':shift}

        self.abs_shift_dict = abs_shift_dict
        
    def reg_xanes2D_chunk(self, overlap_ratio=0.3):
        """
        chunk_sz: int, number of image in one chunk for alignment; each chunk
                  use the last image in that chunk as reference
        method:   str
                  'PC':   skimage.feature.register_translation
                  'MPC':  skimage.feature.masked_register_translation
                  'SR':   pystackreg.StackReg
        overlap_ratio: float, overlap_ratio for method == 'MPC'
        ref_mode: str, control how inter-chunk alignment is done
                  'average': the average of each chunk after intra-chunk
                             re-alignment is used for inter-chunk alignment
                  'single':  the last image in each chunk is used in
                             inter-chunk alignment

        imgs in self.img are registered relative to anchor img. self.img
        is the sub stack with self.img_id_s as its first image, and self.img_id_e
        as the last.
        """
        self.overlap_ratio = overlap_ratio
        self._alignment_scheduler(dtype='2D_XANES')
        self.shifted_image = np.ndarray(self.img.shape)

        if self.img.ndim != 3:
                print('XANES2D image stack is required. Please set XANES2D \
                      image stack first.')
        else:
            if self.method.upper() in {'PC', 'MPC', 'MRTV', 'LS+MRTV', 'MPC+MRTV'}:
                self.shift = np.ndarray([len(self.alignment_pair_list), 2])
            else:
                self.shift = np.ndarray([len(self.alignment_pair_list), 3, 3])

            self.error = np.ndarray(len(self.alignment_pair_list))
            self.si = np.ndarray(len(self.alignment_pair_list))
            self.mse = np.ndarray(len(self.alignment_pair_list))
            self.nrmse = np.ndarray(len(self.alignment_pair_list))

            if self.method.upper() == 'PC':
                print('We are using "phase correlation" method for registration.')
                for ii in range(len(self.alignment_pair_list)):
                    self.shift[ii], self.error[ii], _ = phase_cross_correlation(
                            self.img[self.alignment_pair_list[ii][0]],
                            self.img[self.alignment_pair_list[ii][1]], upsample_factor=100)
                    self.shifted_image[ii] = np.real(np.fft.ifftn(fourier_shift(
                            np.fft.fftn(self.img[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]

            elif self.method.upper() == 'MPC':
                print('We are using "masked phase correlation" method for registration.')
                for ii in range(len(self.alignment_pair_list)):
                    self.shift[ii] = phase_cross_correlation(self.img[self.alignment_pair_list[ii][0]],
                                                             self.img[self.alignment_pair_list[ii][1]],
                                                             reference_mask=self.mask,
                                                             overlap_ratio=self.overlap_ratio)
                    self.shifted_image[ii] = np.real(np.fft.ifftn(fourier_shift(
                            np.fft.fftn(self.img[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]

            elif self.method.upper() == 'SR':
                print('We are using "stack registration" method for registration.')
                if self.mode.upper() == 'TRANSLATION':
                    sr = StackReg(StackReg.TRANSLATION)
                elif  self.mode.upper() == 'RIGID_BODY':
                    sr = StackReg(StackReg.RIGID_BODY)
                elif  self.mode.upper() == 'SCALED_ROTATION':
                    sr = StackReg(StackReg.SCALED_ROTATION)
                elif  self.mode.upper() == 'AFFINE':
                    sr = StackReg(StackReg.AFFINE)
                elif  self.mode.upper() == 'BILINEAR':
                    sr = StackReg(StackReg.BILINEAR)

                if self.mask is not None:
                    for ii in range(len(self.alignment_pair_list)):
                        self.shift[ii] = sr.register(self.img[self.alignment_pair_list[ii][0]]*self.mask,
                                                     self.img[self.alignment_pair_list[ii][1]]*self.mask)
                        self.shifted_image[ii] = sr.transform(self.img[self.alignment_pair_list[ii][1]],
                                                                               tmat=self.shift[ii])[:]
                else:
                    for ii in range(len(self.alignment_pair_list)):
                        self.shift[ii] = sr.register(self.img[self.alignment_pair_list[ii][0]],
                                                     self.img[self.alignment_pair_list[ii][1]])
                        self.shifted_image[ii] = sr.transform(self.img[self.alignment_pair_list[ii][1]],
                                                                               tmat=self.shift[ii])[:]

            elif self.method.upper() == 'MRTV':
                print('We are using "multi-resolution total variation" method for registration.')
                print(self.alg_mrtv_sp_wz, self.alg_mrtv_sp_kernel)
                pxl_conf = {'type': 'area',
                            'levs': self.alg_mrtv_level,
                            'wz': self.alg_mrtv_width,
                            'lsw': 10}
                sub_conf = {'use': True,
                            'type': 'ana',
                            'sp_wz': self.alg_mrtv_sp_wz,
                            'sp_us': 10}

                with mp.get_context('spawn').Pool(self.N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_reg, pxl_conf, sub_conf, None, self.alg_mrtv_sp_kernel),
                                 [[self.img[self.alignment_pair_list[ii][0]],
                                   self.img[self.alignment_pair_list[ii][1]]]
                                   for ii in range(len(self.alignment_pair_list))])
                pool.close()
                pool.join()



                for ii in range(len(rlt)):
                    self.shift[ii] = rlt[ii][3]
                del(rlt)
                gc.collect()

                for ii in range(len(self.alignment_pair_list)):
                    self.shifted_image[ii] = np.real(np.fft.ifftn(fourier_shift(
                            np.fft.fftn(self.img[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]

            elif self.method.upper() == 'MRTV_SINGLE':
                print('We are using "multi-resolution total variation" method for registration.')
                print(self.alg_mrtv_sp_wz, self.alg_mrtv_sp_kernel)
                pxl_conf = {'type': 'area',
                            'levs': self.alg_mrtv_level,
                            'wz': self.alg_mrtv_width,
                            'lsw': 10}
                sub_conf = {'use': True,
                            'type': 'ana',
                            'sp_wz': self.alg_mrtv_sp_wz,
                            'sp_us': 10}
                for a in range(len(self.alignment_pair_list)):
                    _, _, _, tot_shift = mrtv_reg(
                        pxl_conf, sub_conf, None, self.alg_mrtv_sp_kernel,
                        [self.img[self.alignment_pair_list[a][0]],
                         self.img[self.alignment_pair_list[a][1]]]
                    )
                    self.shift[a] = tot_shift
                gc.collect()

                for ii in range(len(self.alignment_pair_list)):
                    self.shifted_image[ii] = np.real(np.fft.ifftn(fourier_shift(
                            np.fft.fftn(self.img[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]


            elif self.method.upper() == 'LS+MRTV':
                print('We are using "line search and multi-resolution total variation" method for registration.')
                with mp.get_context('spawn').Pool(self.N_CPU) as pool:
                    rlt = pool.map(partial(mrtv_ls_combo_reg, self.alg_mrtv_width, 2, 10,
                                           self.alg_mrtv_sp_wz, self.alg_mrtv_sp_wz),
                                   [[self.img[self.alignment_pair_list[ii][0]],
                                     self.img[self.alignment_pair_list[ii][1]]]
                                    for ii in range(len(self.alignment_pair_list))])
                pool.close()
                pool.join()

                for ii in range(len(rlt)):
                    self.shift[ii] = rlt[ii][3]
                del(rlt)
                gc.collect()
                for ii in range(len(self.alignment_pair_list)):
                    self.shifted_image[ii] = np.real(np.fft.ifftn(fourier_shift(
                            np.fft.fftn(self.img[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]

            elif self.method.upper() == 'MPC+MRTV':
                print('We are using combo of "masked phase correlation" and "multi-resolution total variation" method for registration.')
                for ii in range(len(self.alignment_pair_list)):
                    _, _, _, self.shift[ii] = mrtv_mpc_combo_reg(self.img[self.alignment_pair_list[ii][0]],
                                                                 self.img[self.alignment_pair_list[ii][1]],
                                                                 reference_mask=self.mask,
                                                                 overlap_ratio=self.overlap_ratio,
                                                                 levs=self.alg_mrtv_level,
                                                                 wz=self.alg_mrtv_width,
                                                                 sp_wz=self.alg_mrtv_sp_wz,
                                                                 sp_step=self.alg_mrtv_sp_wz)
                    self.shifted_image[ii] = np.real(np.fft.ifftn(fourier_shift(
                            np.fft.fftn(self.img[self.alignment_pair_list[ii][1]]), self.shift[ii])))[:]
        print('Done!')

    def apply_xanes2D_chunk_shift(self, optional_shift_dict, trialfn=None, savefn=None):
        """

        optional_shift_dict: dict; optional
                    user input shifts for specified scan ids. This is useful to
                    correct individual pairs that cannot be aligned with others
                    with the same registration method
        """

        self._sort_absolute_shift(shift_dict=self.shift_dict)

        shift = {}
        for key, item in self.abs_shift_dict.items():
            shift[key] = item['in_sli_shift']

        cnt1 = 0

        for key in sorted(self.abs_shift_dict.keys()):
            shift = self.abs_shift_dict[key]['in_sli_shift']
            self.abs_shifted_image[int(key)] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(self.img[int(key)]), shift)))
            cnt1 += 1

    def _translate_single_img(self, img, shift, method):
        if method.upper() in ['PC', 'MPC', 'MRTV', 'MPC+MRTV']:
            self.img = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(self.img), shift)))
        elif method == 'SR':
            if self.mode.upper() == 'TRANSLATION':
                sr = StackReg(StackReg.TRANSLATION)
            elif  self.mode.upper() == 'RIGID_BODY':
                sr = StackReg(StackReg.RIGID_BODY)
            elif  self.mode.upper() == 'SCALED_ROTATION':
                sr = StackReg(StackReg.SCALED_ROTATION)
            elif  self.mode.upper() == 'AFFINE':
                sr = StackReg(StackReg.AFFINE)
            elif  self.mode.upper() == 'BILINEAR':
                sr = StackReg(StackReg.BILINEAR)
            img[:] = sr.transform(img, tmat = shift)[:]
        else:
            print('Nonrecognized method. Quit!')
            # exit()

    def crop_residual(self):
        abs_shift = self.abs_shift_dict
        sort_keys = sorted(abs_shift)
        shift = np.array([abs_shift[key]['in_sli_shift'] for key in sort_keys])
        shift_py = math.ceil(shift[:, 0].max())
        shift_px = math.ceil(shift[:, 1].max())
        shift_ny = math.floor(shift[:, 0].min())
        shift_nx = math.floor(shift[:, 1].min())
        self.shift_img_cropped = self.abs_shifted_image[:, shift_py:shift_ny, shift_px:shift_nx]
