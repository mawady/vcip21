import os
import imageio
import pywt
import numpy as np

from sklearn import preprocessing
from skimage.transform import resize

import torch
from torch.utils import data
import torchvision.transforms as transforms


class DataLoader(data.Dataset):

    def __init__(self, mode, dataset_input_path, input_size):
        super().__init__()

        self.mode = mode
        self.dataset_input_path = dataset_input_path
        self.input_size = input_size

        self.data, self.labels = self.make_dataset(mode, dataset_input_path)
        self.num_classes = len(np.unique(self.labels))
        print(self.num_classes)

        if len(self.data) == 0:
            raise RuntimeError('Found 0 samples, please check the dataset path')

    def make_dataset(self, mode, path):
        assert mode in ['Train', 'Validation', 'Test']
        if mode is 'Train':
            path = os.path.join(path, 'training_decoded')
        elif mode is 'Validation':
            path = os.path.join(path, 'validation_decoded')
        else:
            path = os.path.join(path, 'test_decoded')

        _files = []
        _labels = []
        subfolders = os.listdir(path)
        for subf in subfolders:
            files = os.listdir(os.path.join(path, subf, 'img'))  # read files of each subfolder
            for f in files:
                _files.append(os.path.join(path, subf, 'img', f))
                _labels.append(subf)

        le = preprocessing.LabelEncoder()
        _labels = le.fit_transform(_labels)
        print(len(_files), len(_labels))

        return _files, _labels

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def __getitem__(self, index):
        img = self.rgb2gray(imageio.imread(self.data[index]))
        cl = self.labels[index]

        c_A_LL, (c_H_LH, c_V_HL, c_D_HH) = pywt.dwt2(img, "haar", mode="reflect")
        if len(c_D_HH.shape) == 2:
            img = np.stack([c_D_HH] * 3, 2)

        # processing each channel separately
        # cR_A_LL, (cR_H_LH, cR_V_HL, cR_D_HH) = pywt.dwt2(img[:, :, 0], "haar", mode="reflect")
        # cG_A_LL, (cG_H_LH, cG_V_HL, cG_D_HH) = pywt.dwt2(img[:, :, 1], "haar", mode="reflect")
        # cB_A_LL, (cB_H_LH, cB_V_HL, cB_D_HH) = pywt.dwt2(img[:, :, 2], "haar", mode="reflect")
        # print(cR_D_HH.shape, cG_D_HH.shape, cB_D_HH.shape)
        # imageio.imwrite('/home/keiller/Desktop/wtd.png', np.concatenate(
        #     (np.expand_dims(cR_D_HH, axis=2), np.expand_dims(cG_D_HH, axis=2), np.expand_dims(cB_D_HH, axis=2)),
        #     axis=-1))
        # img = np.concatenate((np.expand_dims(cR_D_HH, axis=2),
        #                       np.expand_dims(cG_D_HH, axis=2),
        #                       np.expand_dims(cB_D_HH, axis=2)), axis=-1)

        # TODO: data augmentation??

        # normalization
        img = (img - np.min(img))/np.ptp(img)  # normalize 0 and 1
        # img = img.transpose(2, 0, 1)  # change the channel order
        img = resize(img, (self.input_size, self.input_size))

        # https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            # transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(img)

        # Returning to iterator.
        return img.float(), cl

    def __len__(self):
        return len(self.data)
