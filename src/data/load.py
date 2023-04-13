import random

import torch
import torchvision.transforms as transforms

from src.data.ImageNet import (ImageNet_dataset, add_train_labels,
                               get_list_of_files, read_labels)
from src.explanations.augment import augment


def load_train_data(orig_cwd, trained_with_torchvision=False):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32)
    ])
    dict_targets = read_labels(orig_cwd + '/data/meta.mat')
    if trained_with_torchvision:
        pairs = dict_targets.items()
        pairs = sorted(pairs, key=lambda pair: pair[0])
        new_dict = {}
        dict_test = {}
        for i, pair in enumerate(pairs):
            new_dict[pair[0]] = i
            out_test = dict_targets[pair[0]]
            dict_test[out_test] = i
        dict_targets = new_dict

    dir_test = orig_cwd + '/data/imagenet-mini/val/'
    dir_train = orig_cwd + '/data/imagenet-mini/train/'
    train_files = add_train_labels(get_list_of_files(dir_train), dict_targets)
    test_files = add_train_labels(get_list_of_files(dir_test), dict_targets)

    random.shuffle(train_files)
    n_train = int(len(train_files) * 0.9)
    val_files = train_files[n_train:]
    train_files = train_files[:n_train]

    train_dataset = ImageNet_dataset(train_files, transform)
    val_dataset = ImageNet_dataset(val_files, transform)
    test_dataset = ImageNet_dataset(test_files, transform)
    return train_dataset, val_dataset, test_dataset


def augment_data(cfg, aug, min_val, max_val, X_test, post_transform):
    images_aug = []
    values_aug = []
    masks_aug = []

    for image in X_test:
        image_aug, values, masks = augment(image, cfg.n_aug, min_val, max_val,
                                           aug, cfg.seed, cfg.background)
        images_aug.append(torch.stack([post_transform(img_aug) for img_aug in image_aug]))
        values_aug.append(values[1:])
        masks_aug.append(masks)
    return images_aug, values_aug, masks_aug
