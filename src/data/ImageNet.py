import os

import numpy as np
import scipy.io
from PIL import Image
from torch.utils.data import Dataset


def read_labels(path_meta):
    meta = scipy.io.loadmat(path_meta)
    dict_labels = {}
    for i, line in enumerate(meta['synsets'][:1000]):
        dict_labels[list(line[0])[1][0]] = i
    return dict_labels


class dataset_aug(Dataset):
    def __init__(self, dataset, transform_aug, post_transform, should_augment):
        super().__init__()
        self.dataset = dataset
        self.transform_aug = transform_aug
        self.should_augment = should_augment
        self.post_transform = post_transform

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        rand_val = np.random.rand(1)
        try:
            if self.should_augment[label] and rand_val[0] < 0.5:
                image = self.transform_aug(image)
        except Exception as e:
            print(e)
            print(f"Label: {label}, random value: {rand_val}")
        image = self.post_transform(image)
        return image, label

    def __len__(self):
        return self.dataset.__len__()


def get_list_of_files(dir):
    files_list = []
    for path, subdirs, files in os.walk(dir):
        for file in files:
            files_list.append(path + '/' + file)
    return files_list


def add_train_labels(files_list, dict_targets):
    for n, file in enumerate(files_list):
        label = file.split("/")[-2]
        label = dict_targets[label]
        files_list[n] = (file, label)
    return files_list


class ImageNet_dataset(Dataset):
    def __init__(self, files, transform):
        super().__init__()
        self.files = files
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.files[index][0]).convert("RGB")
        image = self.transform(image)
        label = self.files[index][1]
        return image, label

    def __len__(self):
        return len(self.files)
