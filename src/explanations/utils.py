import csv
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from scipy.stats import pearsonr, sem, spearmanr
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate


def correlation(img1: np.ndarray,
                img2: np.ndarray,
                mask: np.ndarray = np.array([[]]),
                type: str = 'pearson',
                ) -> float:
    img1 = img1.flatten()
    img2 = img2.flatten()
    if mask.any():
        if np.max(mask) > len(img1):
            mask = mask[::3] / 3
            mask = mask.astype(np.int32)
        img1 = np.delete(img1, mask)
        img2 = np.delete(img2, mask)
    if type == 'pearson':
        r, p = pearsonr(img1, img2)
    elif type == 'spearman':
        r, p = spearmanr(img1, img2)
    return r


def mean_squared_error(img1: np.ndarray,
                       img2: np.ndarray,
                       mask: np.ndarray = np.array([[]]),
                       ) -> float:
    img1 = img1.flatten()
    img2 = img2.flatten()
    if mask.any():
        if np.max(mask) > len(img1):
            mask = mask[::3] / 3
            mask = mask.astype(np.int32)
        img1 = np.delete(img1, mask)
        img2 = np.delete(img2, mask)
    return mse(img1, img2)


def cos_sim(img1: np.ndarray,
            img2: np.ndarray,
            mask: np.ndarray = np.array([[]]),
            ) -> float:
    img1 = img1.flatten()
    img2 = img2.flatten()
    if mask.any():
        if np.max(mask) > len(img1):
            mask = mask[::3] / 3
            mask = mask.astype(np.int32)
        img1 = np.delete(img1, mask)
        img2 = np.delete(img2, mask)
    img1 = img1.reshape(1, -1)
    img2 = img2.reshape(1, -1)
    return cosine_similarity(img1, img2)[0][0]


def make_mask(image: np.ndarray) -> np.ndarray:
    mask = image == 0
    mask = np.all(mask, axis=-1)
    to_stack = [mask] * image.shape[-1]
    mask = np.stack(to_stack, axis=-1)
    mask = mask.flatten()
    indices = np.argwhere(mask)
    return indices


def accuracy_and_prob(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    softmax = torch.nn.Softmax(dim=1)

    out_soft = softmax(output)
    _, pred = out_soft.topk(maxk, 1, True, True)
    pred = torch.squeeze(pred.t())
    correct = pred.eq(target.expand_as(pred))
    prob = out_soft[0, target]

    res = {}
    for k in topk:
        correct_k = correct[:k].float().sum(0)
        res[f"top-{k}"] = correct_k.numpy()
    return res, prob.numpy()


def topk_intersection(img1, img2, mask, topk):
    img1 = img1.flatten()
    img2 = img2.flatten()
    if mask.any():
        if np.max(mask) > len(img1):
            mask = mask[::3] / 3
            mask = mask.astype(np.int32)
        img1 = np.delete(img1, mask)
        img2 = np.delete(img2, mask)
    _, ind1 = torch.tensor(img1).topk(topk)
    _, ind2 = torch.tensor(img2).topk(topk)
    intersection = [value for value in ind1 if value in ind2]
    return len(intersection)/topk


def save_explanations(images_aug, explanations, masks, img_ids, PF_scores, aug,
                      explainer, cfg):
    aug_dir = cfg.path_expl + aug + '/'
    whole_dir = aug_dir + explainer + '/'
    if not os.path.isdir(aug_dir):
        os.makedirs(aug_dir)
    if not os.path.isdir(whole_dir):
        os.makedirs(whole_dir)
    for i in range(len(img_ids)):
        img_dir = whole_dir + img_ids[i] + '/'
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        expl = [ex.detach().numpy() for ex in explanations[i]]
        np.save(img_dir + 'images_aug.npy', images_aug[i].numpy(), allow_pickle=True)
        np.save(img_dir + 'explanations.npy', expl, allow_pickle=True)
        np.save(img_dir + 'masks.npy', masks[i], allow_pickle=True)
        cfg_dir = img_dir + 'config.yaml'
        with open(cfg_dir, 'w') as f:
            OmegaConf.save(config=cfg, f=f)
        if explainer != "zennit-Occlusion":
            with open(img_dir + 'pixel_flipping.yaml', 'wb') as f:
                pickle.dump(PF_scores[i], f)


def load_explanation_one(folder, explainer):
    images_aug = np.load(folder + 'images_aug.npy', allow_pickle=True)
    explanations = np.load(folder + 'explanations.npy', allow_pickle=True)
    masks = np.load(folder + 'masks.npy', allow_pickle=True)
    with open(folder + 'config.yaml', 'r') as f:
        cfg = OmegaConf.load(f)
    if explainer != "zennit-Occlusion":
        with open(folder + 'pixel_flipping.yaml', 'rb') as f:
            PF_score = pickle.load(f)
    else:
        PF_score = None
    return images_aug, explanations, masks, cfg, PF_score


def load_explanations_all(folder, aug, explainer):
    whole_dir = folder + aug + '/' + explainer + '/'
    subdirs = [x[0] for x in os.walk(whole_dir)]
    images_aug = []
    explanations = []
    masks = []
    cfgs = []
    PF_scores = []
    for folder in subdirs:
        img, expl, mask, cfg, PF_score = load_explanation_one(folder, explainer)
        images_aug.append(img)
        explanations.append(expl)
        masks.append(mask)
        cfgs.append(cfg)
        PF_scores.append(PF_score)
    print(f"Loaded explanations of {len(subdirs)} images for"
          f"augmenation {aug} and explainer {explainer}.")


def unnormalize(data, mean, std):
    std_new = [1/s for s in std]
    mean_new = [-1*m for m in mean]
    invTrans = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0., 0., 0.],
                                                        std=std_new),
                                   transforms.Normalize(mean=mean_new,
                                                        std=[1., 1., 1.]),
                                   transforms.ToPILImage(),
                                   ])

    inv_data = [invTrans(tensor) for tensor in data]
    return inv_data


def load_results(all_results, centre_value=0.0):
    image_ids = {}
    for line in all_results:
        if line["value"] == str(centre_value):
            if line["top-1"] == str(1.0):
                image_ids[line["image_name"]] = 1
            else:
                image_ids[line["image_name"]] = 0
    return image_ids


def save_stats(stats, values, models_to_process, aug, orig_cwd):
    name = f"{orig_cwd}/{aug}-{'-'.join(models_to_process)}-stat.json"
    to_save = {
        "stats": stats,
        "values": values,
    }
    with open(name, 'w') as fout:
        json.dump(to_save, fout)
    return


def load_stats(file_name):
    with open(file_name, 'r') as fin:
        from_saved = json.load(fin)
    stats = from_saved["stats"]
    values = from_saved["values"]
    return stats, values


def write_results(file_name, results):
    with open(file_name, 'w', newline='') as csvfile:
        fieldnames = ['model', 'explainability method', 'class', 'image_name',
                      'augmentation', 'value', 'top-1', 'top-5', 'probability',
                      'correlation_pearson', 'correlation_spearman',
                      'topk_intersection', 'cosine_similarity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for res in results:
            writer.writerow(res)


def prepare_and_write_results(results, acc, probab, values, image_ids,
                              targets, orig_cwd, aug, cfg, labels_dict,
                              model_name):
    results_prepared = []
    for method in results.keys():
        for image in range(len(image_ids)):
            for id in range(len(values[image])):
                res = {
                    "model": cfg.model_name,
                    "explainability method": method,
                    "class": labels_dict[targets[image]],
                    'image_name': image_ids[image],
                    'augmentation': aug,
                    'value': values[image][id],
                    'probability': probab[method][image][id+1],
                    'correlation_pearson': results[method]["correlation_pearson"][image][id],
                    'correlation_spearman': results[method]["correlation_spearman"][image][id],
                    'topk_intersection': results[method]["topk_intersection"][image][id],
                    'cosine_similarity': results[method]["cosine_similarity"][image][id],
                }
                for key in acc[method].keys():
                    res[key] = acc[method][key][image][id+1]
                results_prepared.append(res)
    write_results(orig_cwd + '/outputs/' + model_name + '_'
                  + "ImageNet" + '_' + aug + '_' + method + '.csv',
                  results_prepared)
    return


def fig_to_num(results: List[float],
               values: List[float],
               N: float,
               symmetric: bool = True,
               ) -> float:
    """
    Computes area under a normalized curve for one curve.
    results: list of y-coordinates corresponding to x-coordinates in values (y-axis)
    values: list of numbers that were used as parameters in the augmentation (x-axis)
    N: one of the values, the x-coordinate of an end point of the rectangle
    symmetric: bool indicating whether the values are symmetric with respect to zero
               (e.g. in rotations) or not (e.g. when adding noise)
    """
    N = float(N)
    ind_N = values.index(N)
    if symmetric:
        ind_minN = values.index(min(values, key=lambda x: abs(x-(-1*N))))
        res_cut = results[ind_minN: ind_N+1]
    else:
        res_cut = results[: ind_N + 1]

    diff = 1 - max(res_cut)
    res_cut = [res + diff for res in res_cut]
    ABC = np.trapz(res_cut, dx=1/(len(res_cut)-1))
    return ABC


def fig_to_table(stat: Dict[str, Dict[str, Dict[str, Tuple[List[float]]]]],
                 values: List[float],
                 names: List[str],
                 methods: List[str],
                 N: float = 50,
                 probability_drop=0.0,
                 ) -> None:
    """
    Prints a table  of the criterion with different models in rows and explainability methods
    in columns.
    stat:
    accuracy: dictionary with model names  as keys and accuracies at points "values" as values
    values: list of numbers that were used as parameters in the augmentation (x-axis)
    names: list of names of models
    methods: list of names of explainability methods
    N: one of the values, the x-coordinate of an end point of the rectangle
    """
    table = []
    columns = ["Model"]
    for met in methods:
        columns.append(met)
    symmetric = True
    if values[0] == 0:
        symmetric = False
    results_all = {
        "Correlation Pearson": {},
        "Topk intersection": {},
    }
    for metric in ["Correlation Pearson", "Topk intersection"]:
        results_all[metric] = {}
        for model in names:
            results_all[metric][model] = {}
            tab = [model + ' ratio']
            tab_prob = [model + ' Probability']
            tab_metr = [model + ' ' + metric]
            if probability_drop > 0:
                probabilities = {}
                keys = list(stat[met]["Probability"][model].keys())
                N = keys[-1]
                for key, val in stat[met]["Probability"][model].items():
                    probabilities[key] = np.mean(val)
                zero_index = len(keys)//2
                probab_at_zero = probabilities[keys[zero_index]]
                target_probab = probab_at_zero - probab_at_zero * probability_drop
                for i in range(1, zero_index+1):
                    if (probabilities[keys[zero_index+i]] < target_probab or
                            probabilities[keys[zero_index-i]] < target_probab):
                        print(f"Chosen interval: [{keys[zero_index-i]}, {keys[zero_index+i]}].")
                        print(f"Probability drop from {probab_at_zero} to "
                              f"{min([probabilities[keys[zero_index-i]], probabilities[keys[zero_index+i]]])}.")
                        N = keys[zero_index+i]
                        break

            for met in methods:
                try:
                    normal_factor = [fig_to_num([stat[met]["Probability"][model][str(val)][i]
                                                 for val in values],
                                     values,
                                     N,
                                     symmetric) for i in
                                     range(len(stat[met]["Probability"][model][str(values[0])]))]

                    results = [fig_to_num([stat[met][metric][model][str(val)][i]
                                           for val in values],
                                          values,
                                          N,
                                          symmetric)
                               for i in range(len(stat[met][metric][model][str(values[0])]))]
                except KeyError:
                    normal_factor = [fig_to_num([stat[met]["Probability"][model][val][i]
                                                 for val in values],
                                                values,
                                                N,
                                                symmetric) for i in
                                     range(len(stat[met]["Probability"][model][values[0]]))]

                    results = [fig_to_num([stat[met][metric][model][val][i] for val in values],
                                          values,
                                          N,
                                          symmetric)
                               for i in range(len(stat[met][metric][model][values[0]]))]

                mean_prob, st_em_prob = (np.mean(normal_factor, axis=0),
                                         sem(np.vstack(normal_factor), axis=0))
                mean_metr, st_em_metr = (np.mean(results, axis=0), sem(np.vstack(results), axis=0))
                results = [results[i]/normal_factor[i] for i in range(len(results))]
                results_all[metric][model][met] = results
                mean, st_em = (np.mean(results, axis=0), sem(np.vstack(results), axis=0))
                tab.append(f'{mean:.4}+-{st_em[0]:.4}')
                tab_prob.append(f'{mean_prob:.4}+-{st_em_prob[0]:.4}')
                tab_metr.append(f'{mean_metr:.4}+-{st_em_metr[0]:.4}')
            table.append(tab_prob)
            table.append(tab_metr)
            table.append(tab)
    print(tabulate(table, headers=columns))
    return results_all
