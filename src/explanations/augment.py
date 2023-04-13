from typing import List, Tuple

import numpy as np
import torch
from imgaug import augmenters as iaa

from src.explanations.utils import make_mask


def find_zero_aug(image: np.ndarray, name: str, seed: int, mode: str = 'constant') -> np.ndarray:
    zero_aug = {
        "AddToBrightness": iaa.AddToBrightness(0,
                                               from_colorspace='RGB',
                                               seed=seed).to_deterministic(),
        "AddToHue": iaa.AddToHue(0, from_colorspace='RGB', seed=seed).to_deterministic(),
        "AddToSaturation": iaa.AddToSaturation(0,
                                               from_colorspace='RGB',
                                               seed=seed).to_deterministic(),
        "AdditiveGaussianNoise": iaa.AdditiveGaussianNoise(scale=0,
                                                           seed=seed).to_deterministic(),
        "Rotate": iaa.Rotate(rotate=0.,
                             seed=seed,
                             mode=mode,
                             ).to_deterministic(),
        "Grayscale": iaa.Grayscale(alpha=0.0,
                                   from_colorspace='RGB',
                                   seed=seed).to_deterministic(),
        "ChangeColorspace": iaa.ChangeColorspace('RGB',
                                                 from_colorspace='RGB',
                                                 alpha=0.0,
                                                 seed=seed).to_deterministic(),
        "Dropout": iaa.Dropout(0.0, seed=seed).to_deterministic(),
        "TranslateX": iaa.TranslateX(percent=0.0,
                                     seed=seed,
                                     mode=mode).to_deterministic(),
        "TranslateY": iaa.TranslateY(percent=0.0,
                                     seed=seed,
                                     mode=mode).to_deterministic(),
        "Translate": iaa.Affine(translate_percent=0.0,
                                seed=seed,
                                mode=mode).to_deterministic(),
        "Scale": iaa.Affine(scale=1.0,
                            seed=seed,
                            mode=mode).to_deterministic(),
        "GaussianBlur": iaa.GaussianBlur(sigma=0.0,
                                         seed=seed).to_deterministic()
    }
    aug_func = zero_aug[name]
    if len(image.shape) == 4:
        new_im = aug_func(images=image)
    else:
        new_im = aug_func(image=image)
    mask = make_mask(new_im)
    return new_im, mask


def augment(image: np.ndarray,
            n: int,
            min: int,
            max: int,
            name: str,
            seed: int,
            mode: str = 'constant',
            ) -> Tuple[np.ndarray, List[float]]:
    is_float = False
    if image.max() <= 1:
        is_float = True
        image = image * 255
        image = np.uint8(image)
    k = (max - min) / n
    img_aug = []
    values = []
    masks = []
    moveaxis = False
    if isinstance(image, torch.Tensor):
        image = np.moveaxis(image.numpy(), 0, 2)
        moveaxis = True
    elif image.shape[0] < 4:
        image = np.moveaxis(image, 0, 2)
        moveaxis = True
    img, mask = find_zero_aug(image, name, seed, mode)
    if len(image.shape) == 4:
        img_aug.append(img)
    else:
        img_aug.append(np.expand_dims(img, 0))
    masks.append(mask)
    values.append(0.)
    for i in range(n+1):
        if name == "AddToBrightness":
            aug_func = iaa.AddToBrightness(min + i*k,
                                           from_colorspace='RGB',
                                           seed=seed,
                                           ).to_deterministic()
            aug_func_mask = aug_func
        elif name == "AddToHue":
            aug_func = iaa.AddToHue(int(min + i*k),
                                    from_colorspace='RGB',
                                    seed=seed).to_deterministic()
            aug_func_mask = aug_func
        elif name == "AddToSaturation":
            aug_func = iaa.AddToSaturation(int(min + i * k),
                                           from_colorspace='RGB',
                                           seed=seed).to_deterministic()
            aug_func_mask = aug_func
        elif name == "AdditiveGaussianNoise":
            aug_func = iaa.AdditiveGaussianNoise(scale=min + i*k,
                                                 seed=seed).to_deterministic()
            aug_func_mask = aug_func
        elif name == "Rotate":
            aug_func = iaa.Rotate(rotate=min + i*k,
                                  seed=seed,
                                  mode=mode).to_deterministic()
            aug_func_mask = iaa.Rotate(rotate=min + i*k,
                                       seed=seed,
                                       mode="constant").to_deterministic()
        elif name == "Grayscale":
            aug_func = iaa.Grayscale(alpha=min + i*k,
                                     from_colorspace='RGB',
                                     seed=seed).to_deterministic()
            aug_func_mask = aug_func
        elif name == "ChangeColorspace":
            aug_func = iaa.ChangeColorspace('RGB',
                                            from_colorspace='RGB',
                                            alpha=min + i*k,
                                            seed=seed,
                                            ).to_deterministic()
            aug_func_mask = aug_func
        elif name == "Dropout":
            aug_func = iaa.Dropout(min + i*k,
                                   seed=seed).to_deterministic()
            aug_func_mask = aug_func
        elif name == "TranslateX":  # 0 denotes “no change” and 0.5 denotes “half of the axis size”
            aug_func = iaa.TranslateX(percent=min + i*k,
                                      seed=seed,
                                      mode=mode).to_deterministic()
            aug_func_mask = iaa.TranslateX(percent=min + i*k,
                                           seed=seed,
                                           mode="constant").to_deterministic()
        elif name == "TranslateY":  # 0 denotes “no change” and 0.5 denotes “half of the axis size”
            aug_func = iaa.TranslateY(percent=min + i * k,
                                      seed=seed,
                                      mode=mode).to_deterministic()
            aug_func_mask = iaa.TranslateY(percent=min + i * k,
                                           seed=seed,
                                           mode="constant").to_deterministic()
        elif name == "Translate":
            aug_func = iaa.Affine(translate_percent=min + i * k,
                                  seed=seed,
                                  mode=mode).to_deterministic()
            aug_func_mask = iaa.Affine(translate_percent=min + i * k,
                                       seed=seed,
                                       mode="constant").to_deterministic()
        elif name == "Scale":  # Scaling factor to use, where 1.0 denotes “no change”
            # and 0.5 is zoomed out to 50 percent of the original size.
            aug_func = iaa.Affine(scale=min + i*k,
                                  seed=seed,
                                  mode=mode).to_deterministic()
            aug_func_mask = iaa.Affine(scale=min + i*k,
                                       seed=seed,
                                       mode="constant").to_deterministic()
        elif name == "GaussianBlur":  # Values in the range 0.0 (no blur) to 3.0 (strong blur)
            aug_func = iaa.GaussianBlur(sigma=min + i*k,
                                        seed=seed).to_deterministic()
            aug_func_mask = aug_func
        else:
            raise ValueError("Unknown augmentation method.")
        values.append(min + i * k)
        if len(image.shape) == 4:
            new_im = aug_func(images=image)
            mask = aug_func_mask(images=np.ones_like(image))
            img_aug.append(new_im)
        else:
            new_im = aug_func(image=image)
            mask = aug_func_mask(image=np.ones_like(image)*255)
            img_aug.append(np.expand_dims(new_im, 0))
        masks.append(make_mask(mask))
    images_aug = np.concatenate(img_aug, axis=0)
    if is_float:
        images_aug = images_aug / 255
        images_aug = np.float32(images_aug)
    if moveaxis:
        images_aug = np.moveaxis(images_aug, -1, 1)
    images_aug = torch.from_numpy(images_aug)
    return images_aug, values, masks


def augment_explanations(image: np.ndarray, name: str, mode: str, seed: int, values) -> np.ndarray:
    if name == "Rotate":
        new_images = []
        for i in range(len(values)):
            aug_func = iaa.Rotate(rotate=values[i],
                                  mode=mode,
                                  seed=seed).to_deterministic()
            new_images.append(aug_func(image=image))
        images = new_images
    elif name == "Scale":
        new_images = []
        for i in range(len(values)):
            aug_func = iaa.Affine(scale=values[i],
                                  seed=seed,
                                  mode=mode).to_deterministic()
            new_images.append(aug_func(image=image))
        images = new_images
    elif name == "TranslateX":
        new_images = []
        for i in range(len(values)):
            aug_func = iaa.TranslateX(percent=values[i],
                                      seed=seed,
                                      mode=mode).to_deterministic()
            new_images.append(aug_func(image=image))
        images = new_images
    elif name == "TranslateY":
        new_images = []
        for i in range(len(values)):
            aug_func = iaa.TranslateY(percent=values[i],
                                      seed=seed,
                                      mode=mode).to_deterministic()
            new_images.append(aug_func(image=image))
        images = new_images
    elif name == "Translate":
        new_images = []
        for i in range(len(values)):
            aug_func = iaa.Affine(translate_percent=values[i],
                                  seed=seed,
                                  mode=mode).to_deterministic()
            new_images.append(aug_func(image=image))
        images = new_images
    elif name == "Dropout":
        new_images = []
        for i in range(len(values)):
            aug_func = iaa.Dropout(values[i],
                                   seed=seed).to_deterministic()
            new_images.append(aug_func(image=image))
        images = new_images
    else:
        images = len(values) * [image]
    return images
