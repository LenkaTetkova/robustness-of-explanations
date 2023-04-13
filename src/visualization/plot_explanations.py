import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from src.explanations.utils import load_explanation_one, unnormalize
from src.visualization.utils import make_cmap


def find_and_make_dirs(img_id, aug, explainer, old, new):
    aug_dir = aug + '/'
    whole_dir = aug_dir + explainer + '/'
    if not os.path.isdir(new + aug_dir):
        os.makedirs(new + aug_dir)
    if not os.path.isdir(new + whole_dir):
        os.makedirs(new + whole_dir)
    img_dir = whole_dir + img_id + '/'
    if not os.path.isdir(new + img_dir):
        os.makedirs(new + img_dir)
    old_path = old + img_dir
    new_path = new + img_dir
    return old_path, new_path


def make_mask(mask, shape):
    img_reshap = np.zeros(shape)
    img_reshap = img_reshap.flatten()
    for ind in mask:
        img_reshap[ind] = 1
    img_new = np.reshape(img_reshap, shape)
    return img_new


def get_min_max(img):
    minval = np.min(img)
    maxval = np.max(img)
    abs_val = np.max([np.abs(minval), np.abs(maxval)])
    return -1*abs_val, abs_val


def save_expl_plots(expl, color_map, path, mask, normalize):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    expl_masked = expl
    mask = np.mean(mask, axis=-1)
    masked = np.ma.masked_where(mask == 0, mask)
    if normalize == 'max':
        _, norm_factor = get_min_max(expl_masked)
    elif normalize == 'p99':
        norm_factor = np.percentile(np.abs(expl_masked), 99)
    else:
        norm_factor = 1
    expl_all = np.mean(expl_masked, axis=-1) / norm_factor
    ax.imshow(expl_all, cmap=color_map, vmin=-1, vmax=1)
    ax.imshow(masked, cmap='binary', alpha=1, vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(f"All channels - {normalize}")
    fig.savefig(path + ".png", bbox_inches='tight')
    plt.close(fig)

    fig, axs = plt.subplots(1, 3)
    for i in range(3):
        if normalize == 'max':
            _, norm_factor = get_min_max(expl_masked[:, :, i])
        elif normalize == 'p99':
            norm_factor = np.percentile(np.abs(expl_masked[:, :, i]), 99)
        else:
            norm_factor = 1
        expl_single = expl_masked[:, :, i] / norm_factor
        axs[i].imshow(expl_single, cmap=color_map, vmin=-1, vmax=1)
        axs[i].imshow(masked, cmap='binary', alpha=1, vmin=0, vmax=1)
        axs[i].axis("off")
        axs[i].set_title(f"Channel {i} - {normalize}")
    fig.savefig(path + "_ch.png", bbox_inches='tight')
    plt.close(fig)


def plot_and_save_explanations(images_aug, explanations, masks, cfg, new_path, aug, color_map,
                               normalize):
    images_aug = np.moveaxis(images_aug, 1, -1)
    explanations = np.moveaxis(explanations, 1, -1)
    min_val, max_val = cfg.augmentations[aug]
    n = len(images_aug) - 2
    k = (max_val - min_val) / n
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images_to_plot = unnormalize(images_aug, mean, std)
    for i in range(n+1):
        value = min_val + i*k
        images_to_plot[i+1].save(f"{new_path}Image_{value}.png")
        mask = make_mask(masks[i+1], explanations[i+1].shape)
        save_expl_plots(explanations[i+1], color_map, f"{new_path}Expl_{value}", mask,
                        normalize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug", nargs="+",
                        help="Augmentation type(s)")
    parser.add_argument("--explainer", nargs="+",
                        help="Explainer type(s)")
    parser.add_argument("folder", type=str,
                        help="Root folder where the explanations are saved.")
    parser.add_argument("image_id", type=str,
                        help="ID if the image to plot explanations.")
    parser.add_argument("folder_to_save", type=str,
                        help="Root folder where the figures will be saved.")
    parser.add_argument("--normalize", type=str, default='max',
                        help="Way of normalizing the explanations."
                             "Currently either 'max' or 'p99'.")

    args = parser.parse_args()

    color_map = make_cmap()

    for aug in args.aug:
        for explainer in args.explainer:
            old_path, new_path = find_and_make_dirs(args.image_id,
                                                    aug,
                                                    explainer,
                                                    args.folder,
                                                    args.folder_to_save,
                                                    )
            images_aug, explanations, masks, cfg, _ = load_explanation_one(old_path, explainer)
            plot_and_save_explanations(images_aug, explanations, masks, cfg,
                                       new_path, aug, color_map, args.normalize)
