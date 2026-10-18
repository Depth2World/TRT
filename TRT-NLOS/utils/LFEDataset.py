import os
import glob
TRUE = True
import numpy as np
import cv2
from torch.utils.data import Dataset
import scipy.io as sio
import torch


def list_file_path_bike(root_path, fortrain=TRUE, shineness=[0]):
    modeldirs = []
    test_modeldirs = []
    mea = []
    im = []
    de = []
    test_mea = []
    test_im = []
    test_de = []
    for fol in root_path:
        for shi in shineness:
            modeldir_all = glob.glob('%s/%d/*' % (fol, shi))  # file_path/0/XXX
            modeldir_all = sorted(modeldir_all)  # sort
            modeldir_all.reverse()


            for modeldir in modeldir_all[:250]:
                rotdirs = glob.glob('%s/shin*' % (modeldir))  # file_path/0/XXX/shinexxx_rot xxx
                rotdirs = sorted(rotdirs)
                rotdirs.reverse()

                modeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspad*.mp4' % (dir))
                    mea.append(path)
                    path = glob.glob('%s/confocal-0-*.hdr' % (dir))
                    im.append(path)
                    path = glob.glob('%s/depth-0-*.hdr' % (dir))
                    de.append(path)

            for modeldir in modeldir_all[250:]:
                rotdirs = glob.glob('%s/shin*' % (modeldir))  # [:1]   # file_path/0/XXX/shinexxx_rot xxx
                rotdirs = sorted(rotdirs)
                rotdirs.reverse()
                test_modeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocalspad*.mp4' % (dir))
                    test_mea.append(path)
                    path = glob.glob('%s/confocal-0-*.hdr' % (dir))
                    test_im.append(path)
                    path = glob.glob('%s/depth-0-*.hdr' % (dir))
                    test_de.append(path)

    train_sample = {'Mea': mea, 'dep': de, 'img': im, 'path': modeldirs}
    test_sample = {'Mea': test_mea, 'dep': test_de, 'img': test_im, 'path': test_modeldirs}
    if fortrain:
        return train_sample
    else:
        return test_sample


def check_file(path):
    if not os.path.isfile(path):
        raise ValueError('file does not exist: %s' % path)


class LFEDataset(Dataset):

    def __init__(
            self,
            root,  # dataset root directory
            shineness,  # data split ('train', 'val')
            for_train=True,
            ds=1,  # temporal down-sampling factor
            clip=512,  # time range of histograms
            size=256,  # measurement size (unit: px)
            scale=1,  # scaling factor (float or float tuple)
            background=0,  # background noise rate (float or float tuple)
            target_size=128,  # target image size (unit: px)
            target_noise=0.01,  # standard deviation of target image noise
            color='gray', clean_measurement_pattern=None):

        super(LFEDataset, self).__init__()
        self.root = root
        self.clean_measurement_pattern = clean_measurement_pattern
        self.ds = ds
        self.clip = clip
        self.size = size
        self.target_size = target_size
        self.target_noise = target_noise
        assert color in ('rgb', 'gray', 'r', 'g', 'b'), \
            'invalid color: {:s}'.format(color)
        self.data_list = list_file_path_bike(root, for_train, shineness)
        self.color = color
        for key in ('Mea', 'img', 'dep'):
            for paths in self.data_list[key]:
                if len(paths) != 1:
                    raise ValueError(f'Expected one {key} file per view, got {paths}')

    def _load_meas(self, idx, path=None):
        if path is None:
            path = self.data_list['Mea'][idx][0]
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr', 'mp4')
        try:
            if ext == 'mat':
                raw = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            elif ext == 'hdr':
                raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                raw = raw.reshape(-1, raw.shape[1], raw.shape[1], 1)
                raw = raw.transpose(3, 0, 1, 2)  # 1 600 256 256
            else:
                cap = cv2.VideoCapture(path)
                assert cap.isOpened()
                ims = []
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        ims.append(frame)
                    else:
                        break
                cap.release()
                raw = np.array(ims, dtype=np.float32) / 255.0 # normalize
                raw = raw.transpose(3, 0, 1, 2)  # c 600 256 256
                if (self.size/self.target_size ==2 ):
                    raw = raw[:, :, ::2, :] + raw[:, :, 1::2, :]
                    raw = raw[:, :, :, ::2] + raw[:, :, :, 1::2]
            if raw.ndim == 3:
                raw = raw[None]
            if self.ds > 1:
                c, t, h, w = raw.shape
                assert t % self.ds == 0
                raw = raw.reshape(c, t // self.ds, self.ds, h, w)
                raw = raw.sum(axis=2)
            raw = raw[:, :self.clip]
            if raw.shape[1:] != (self.clip, self.target_size, self.target_size):
                raise ValueError(f'Unexpected transient shape: {raw.shape}')

            if raw.shape[0] == 3:
                if self.color == 'gray':
                    raw = 0.299 * raw[0:1] + 0.587 * raw[1:2] + 0.114 * raw[2:3]
                elif self.color == 'r':
                    raw = raw[0:1]
                elif self.color == 'g':
                    raw = raw[1:2]
                elif self.color == 'b':
                    raw = raw[2:3]
            else:
                if self.color == 'rgb':
                    raw = np.tile(raw, [3, 1, 1, 1])
        except Exception as exc:
            raise ValueError(f'Measurement loading failed: {path}: {exc}') from exc


        c, t, h, w = raw.shape
        raw = torch.from_numpy(raw.astype(np.float32))  # (1/3, t, h, w)
        return raw

    def _load_image(self, idx):
        path = self.data_list['img'][idx][0]
        check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr', 'png', 'jpg', 'jpeg')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                x = x.astype(np.float32)
                if ext == 'hdr':
                    x = x.reshape(x.shape[1], x.shape[1], 1)  # 256 256 1 1
                    x = cv2.resize(x, (self.target_size, self.target_size))
                    x = x.reshape(1, x.shape[1], x.shape[1])  # 128 128 1 1
                    x = x / max(float(np.max(x)), 1e-8)
                else:
                    x = x / 255
        except:
            raise ValueError('image loading failed: {:s}'.format(path))

        x = torch.from_numpy(x.astype(np.float32))  # (h, w, v, 1/3)       #  256 256 1 1
        return x

    def _load_depth(self, idx):
        path = self.data_list['dep'][idx][0]
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # meters  fe 0.597m
                x = x[..., 0]
                if x.ndim == 3:
                    x = x[..., 0]
                x = x.clip(0, 1)
            x = cv2.resize(x, (self.target_size, self.target_size))  # (h, w)
        except:
            raise ValueError('depth loading failed: {:s}'.format(path))
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)

        x = x.transpose(2, 3, 0, 1)
        x = torch.from_numpy(x.astype(np.float32))  # (v, 1, h, w)
        return x[0]

    def __len__(self):
        return len(self.data_list['Mea'])

    def __getitem__(self, idx):
        meas = self._load_meas(idx)
        images = self._load_image(idx)
        depths = self._load_depth(idx)
        sample = {'ds_meas': meas, 'dep_gt': depths, 'img_gt': images}
        if self.clean_measurement_pattern:
            directory = os.path.dirname(self.data_list['Mea'][idx][0])
            matches = sorted(glob.glob(os.path.join(directory, self.clean_measurement_pattern)))
            if len(matches) != 1:
                raise ValueError(f'Expected one clean transient in {directory}, got {matches}')
            if os.path.realpath(matches[0]) == os.path.realpath(self.data_list['Mea'][idx][0]):
                raise ValueError('Clean supervision must not be the noisy input file.')
            sample['raw_meas'] = self._load_meas(idx, matches[0])
            if sample['raw_meas'].shape != meas.shape:
                raise ValueError('Clean and noisy transient shapes must match.')
        return sample


class LFEDataset_unseen(Dataset):

    def __init__(
            self,
            root,  # dataset root directory
            shineness,  # data split ('train', 'val')
            for_train=True,
            ds=1,  # temporal down-sampling factor
            clip=512,  # time range of histograms
            size=128,  # measurement size (unit: px)
            scale=1,  # scaling factor (float or float tuple)
            background=0,  # background noise rate (float or float tuple)
            target_size=128,  # target image size (unit: px)
            target_noise=0.01,  # standard deviation of target image noise
            color='gray',  # color channel(s) of target image
            sp_ds=8,  # spatial resolution downsample,
            mask=False
    ):
        super(LFEDataset_unseen, self).__init__()
        self.root = root
        self.ds = ds
        self.clip = clip
        self.size = size
        self.target_size = target_size
        self.target_noise = target_noise
        assert color in ('rgb', 'gray', 'r', 'g', 'b'), \
            'invalid color: {:s}'.format(color)
        self.data_list = list_file_unseen(root, for_train, shineness)
        self.color = color
        for key in ('Mea', 'img', 'dep'):
            for paths in self.data_list[key]:
                if len(paths) != 1:
                    raise ValueError(f'Expected one {key} file per view, got {paths}')
        self.sp_ds = sp_ds
        self.mask = mask


    def _load_meas(self, idx):
        path = self.data_list['Mea'][idx][0]
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr', 'mp4')
        try:
            if ext == 'mat':
                raw = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            elif ext == 'hdr':
                raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                raw = raw.reshape(-1, raw.shape[1], raw.shape[1], 1)
                raw = raw.transpose(3, 0, 1, 2)  # 1 600 256 256
            else:
                cap = cv2.VideoCapture(path)
                assert cap.isOpened()
                ims = []
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        ims.append(frame)
                    else:
                        break
                cap.release()
                raw = np.array(ims, dtype=np.float32) / 255.0 # 归一化
                raw = raw.transpose(3, 0, 1, 2)  # 3 600 256 256
            if self.size / self.target_size == 2:
                raw = raw[:, :, ::2, :] + raw[:, :, 1::2, :]
                raw = raw[:, :, :, ::2] + raw[:, :, :, 1::2]
            if raw.ndim == 3:
                raw = raw[None]

            if self.ds > 1:
                c, t, h, w = raw.shape
                assert t % self.ds == 0
                raw = raw.reshape(c, t // self.ds, self.ds, h, w)
                raw = raw.sum(axis=2)

            raw = raw[:, :self.clip]
            if raw.shape[1:] != (self.clip, self.target_size, self.target_size):
                raise ValueError(f'Unexpected transient shape: {raw.shape}')
            if raw.shape[0] == 3:
                if self.color == 'gray':
                    raw = 0.299 * raw[0:1] + 0.587 * raw[1:2] + 0.114 * raw[2:3]
                elif self.color == 'r':
                    raw = raw[0:1]
                elif self.color == 'g':
                    raw = raw[1:2]
                elif self.color == 'b':
                    raw = raw[2:3]
            else:
                if self.color == 'rgb':
                    raw = np.tile(raw, [3, 1, 1, 1])
        except Exception as exc:
            raise ValueError(f'Measurement loading failed: {path}: {exc}') from exc


        c, t, h, w = raw.shape
        x = torch.from_numpy(raw.astype(np.float32))  # (1/3, t, h, w)
        raw = x
        return x, raw

    def _load_image(self, idx):
        path = self.data_list['img'][idx][0]
        check_file(path)
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr', 'png', 'jpg', 'jpeg')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                x = x.astype(np.float32)
                if ext == 'hdr':
                    x = x.reshape(x.shape[1], x.shape[1], 1)  # 256 256 1 1
                    x = cv2.resize(x, (self.target_size, self.target_size))
                    x = x.reshape(1, x.shape[1], x.shape[1])  # 128 128 1 1
                    x = x / max(float(np.max(x)), 1e-8)
                else:
                    x = x / 255
        except:
            raise ValueError('image loading failed: {:s}'.format(path))

        x = torch.from_numpy(x.astype(np.float32))  # (h, w, v, 1/3)       #  256 256 1 1
        return x.reshape(1, self.target_size, self.target_size)

    def _load_depth(self, idx):
        path = self.data_list['dep'][idx][0]
        ext = path.split('.')[-1]
        assert ext in ('mat', 'hdr', 'png')
        try:
            if ext == 'mat':
                x = sio.loadmat(
                    path, verify_compressed_data_integrity=False
                )['data']
            else:
                x = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # meters  fe 0.597m
                if x.ndim == 3:
                    x = x[..., 0]
                x = x.clip(0, 1)
            x = cv2.resize(x, (self.target_size, self.target_size))  # (h, w)
        except:
            raise ValueError('depth loading failed: {:s}'.format(path))
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)

        x = x.transpose(2, 3, 0, 1)
        x = torch.from_numpy(x.astype(np.float32))  # (v, 1, h, w)
        return x[0]

    def __len__(self):
        return len(self.data_list['Mea'])

    def __getitem__(self, idx):
        meas, raw_meas = self._load_meas(idx)  # self.transform()  #   the hyperparameter [0.05 2] has err
        images = self._load_image(idx)
        depths = self._load_depth(idx)
        sample = {'ds_meas': meas, 'raw_meas': raw_meas, 'dep_gt': depths, 'img_gt': images}
        return sample


def list_file_unseen(root_path, fortrain=TRUE, shineness=[0]):
    modeldirs = []
    mea = []
    im = []
    de = []
    for fol in root_path:
        for shi in shineness:
            modeldir_all = glob.glob('%s/%d/*' % (fol, shi))  # file_path/0/XXX
            modeldir_all = sorted(modeldir_all)  # 字典序排序
            modeldir_all.reverse()

            for modeldir in modeldir_all:
                rotdirs = glob.glob('%s/shin*' % (modeldir))  # file_path/0/XXX/shinexxx_rot xxx
                rotdirs = sorted(rotdirs)
                rotdirs.reverse()
                modeldirs.extend(rotdirs)
                for dir in rotdirs:
                    path = glob.glob('%s/video-confocal-gray-full*.mp4' % (dir))
                    if len(path)==0:continue
                    mea.append(path)
                    path = glob.glob('%s/confocal-0*.png' % (dir))
                    im.append(path)
                    path = glob.glob('%s/depth-0*.png' % (dir))
                    de.append(path)

    train_sample = {'Mea': mea, 'dep': de, 'img': im, 'path': modeldirs}
    return train_sample

