import random
import time

import hydra
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from omegaconf import DictConfig, OmegaConf

from src.data.ImageNet import read_labels
from src.data.load import augment_data, load_train_data
from src.explanations.augment import augment_explanations
from src.explanations.explain_captum import explain_captum
from src.explanations.utils import (correlation, cos_sim,
                                    prepare_and_write_results,
                                    save_explanations, topk_intersection)
from src.models.torch_model import get_post_transform, load_model


def compute_results(explanations, masks, seed, values, aug, background,
                    folder):
    results = {}
    for method, expl in explanations.items():
        augmented_zero = [augment_explanations(
            np.moveaxis(expl[i][0].clone().detach().numpy(), 0, -1),
            aug, background, seed, values[i])
                          for i in range(len(expl))]
        expl = [[np.moveaxis(img.clone().detach().numpy(), 0, - 1) for img in images] for images in
                expl]
        corr_pearson_val = [[correlation(augmented_zero[i][j - 1],
                                         expl[i][j], masks[i][j], type='pearson')
                             for j in range(1, len(expl[i]))] for i in range(len(expl))]
        corr_spearman_val = [[correlation(augmented_zero[i][j - 1],
                                          expl[i][j], masks[i][j], type='spearman')
                             for j in range(1, len(expl[i]))] for i in range(len(expl))]

        topk_val = [[topk_intersection(augmented_zero[i][j - 1],
                                       expl[i][j],
                                       masks[i][j],
                                       1000)
                    for j in range(1, len(expl[i]))] for i in range(len(expl))]
        cos_sim_val = [[cos_sim(augmented_zero[i][j - 1],
                                expl[i][j], masks[i][j])
                        for j in range(1, len(expl[i]))] for i in range(len(expl))]
        results[method] = {
            "correlation_pearson": corr_pearson_val,
            "correlation_spearman": corr_spearman_val,
            "topk_intersection": topk_val,
            "cosine_similarity": cos_sim_val,
        }
    return results


@hydra.main(config_path="../config", config_name="default.yaml")
def compare_explanations(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()

    # Fix random seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    model, layer = load_model(cfg.model_name, cfg.pretrained,
                              orig_cwd + cfg.checkpoint_path,
                              use_model_ema=cfg.use_model_ema)
    model.eval()
    model.to(device)

    # Load data
    _, _, test_dataset = load_train_data(orig_cwd,
                                         cfg.trained_with_torchvision)
    indices = np.random.randint(0, len(test_dataset), size=cfg.n_to_expl)
    images = [test_dataset.__getitem__(ind)[0] for ind in indices]
    img_ids = [test_dataset.files[ind][0][-28:-5] for ind in indices]
    n_classes = 1000
    targets = [test_dataset.__getitem__(ind)[1] for ind in indices]
    post_transform = get_post_transform()

    for aug, (min_value, max_value) in cfg.augmentations.items():
        (images_aug, values, masks) = augment_data(cfg, aug,
                                                   min_value, max_value,
                                                   images, post_transform)
        labels_dict = {}
        dict_targets = read_labels(orig_cwd + '/data/meta.mat')
        if cfg.use_model_ema:
            pairs = dict_targets.items()
            pairs = sorted(pairs, key=lambda pair: pair[0])
            new_dict = {}
            for i, pair in enumerate(pairs):
                new_dict[pair[0]] = i
            dict_targets = new_dict
        for key, value in dict_targets.items():
            labels_dict[value] = key

        for explainer in cfg.expl_methods:
            start = time.time()
            explanations, acc, probab, PF_scores = explain_captum(model,
                                                                  images_aug,
                                                                  targets,
                                                                  explainer,
                                                                  device,
                                                                  masks,
                                                                  orig_cwd,
                                                                  model_layer=layer,
                                                                  model_name=cfg.model_name,
                                                                  n_classes=n_classes,
                                                                  )
            if cfg.save_expl:
                save_explanations(images_aug, explanations[explainer], masks, img_ids,
                                  PF_scores, aug, explainer, cfg)
            results = compute_results(explanations, masks, cfg.seed, values, aug,
                                      cfg.background, orig_cwd)
            model_name = cfg.checkpoint_path.split("/")[-1][:-4]
            prepare_and_write_results(results, acc, probab, values, img_ids, targets,
                                      orig_cwd, aug, cfg, labels_dict, model_name)
            print(time.strftime("%H:%M:%S +0000", time.gmtime(time.time() - start)))


if __name__ == "__main__":
    compare_explanations()
