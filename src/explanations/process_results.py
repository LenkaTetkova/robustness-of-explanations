import csv
from copy import deepcopy

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.stats import sem

from src.explanations.utils import fig_to_table, load_results, save_stats
from src.visualization.visualize import (visualise_acc, visualise_expl,
                                         visualise_probab)


def read_results(file_name):
    rows = []
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    return rows


def sort_by_img_id(results_all, aug, only_correct=False):
    new_results = {}
    for key, results in results_all.items():
        results_sorted = {}
        if only_correct:
            if aug == "Scale":
                is_correct = load_results(results, centre_value=1.0)
            else:
                is_correct = load_results(results, centre_value=0.0)
        expl_methods = set([res["explainability method"] for res in results])
        image_ids = set([res["image_name"] for res in results])
        for method in expl_methods:
            results_sorted[method] = {}
            for image in image_ids:
                if not only_correct or is_correct[image]:
                    results_sorted[method][image] = []
        for row in results:
            if not only_correct or is_correct[row["image_name"]]:
                results_sorted[row["explainability method"]][row["image_name"]].append(row)
        new_results[key] = results_sorted
    return new_results


def aggregate_and_visualise(results_all, aug, min_value, max_value, orig_cwd, N,
                            models_to_process, probability_drop=0.0):
    stat = {}
    stat_for_table = {}
    dict_acc = {}
    prob = {}
    for model_name, model_results in results_all.items():
        for method in model_results.keys():
            stat_for_table[method] = {
                "Correlation Pearson": {},
                "Correlation Spearman": {},
                "Topk intersection": {},
                "Probability": {},
                "Top 1": {},
                "Top 5": {},
            }
    for model_name, model_results in results_all.items():
        stat[model_name] = {
                             "Correlation Pearson": {},
                             "Correlation Spearman": {},
                             "Topk intersection": {},
                             }
        for method, results in model_results.items():
            probabs = {}
            top1 = {}
            top5 = {}
            correlation_p = {}
            correlation_s = {}
            topk_intersection = {}
            prob[model_name] = {}
            values_all = [[float(results[image][i]["value"])
                           for i in range(len(results[image]))]
                          for image in results.keys()]
            values = set.intersection(*map(set, values_all))
            values = [val if (min_value <= val <= max_value) else 0 for val in values]
            values = sorted(list(set(values)))
            for val in values:
                probabs[val] = []
                top1[val] = []
                top5[val] = []
                correlation_p[val] = []
                correlation_s[val] = []
                topk_intersection[val] = []
                is_top5 = False
            for name, image in results.items():
                for img in image:
                    val = float(img["value"])
                    probabs[val].append(float(img["probability"]))
                    top1[val].append(float(img["top-1"]))
                    if img["top-5"]:
                        top5[val].append(float(img["top-5"]))
                        is_top5 = True
                    correlation_p[val].append(float(img["correlation_pearson"]))
                    correlation_s[val].append(float(img["correlation_spearman"]))
                    topk_intersection[val].append(float(img["topk_intersection"]))
            stat_for_table[method]["Correlation Pearson"][model_name] = deepcopy(correlation_p)
            stat_for_table[method]["Correlation Spearman"][model_name] = deepcopy(correlation_s)
            stat_for_table[method]["Topk intersection"][model_name] = deepcopy(topk_intersection)
            stat_for_table[method]["Probability"][model_name] = deepcopy(probabs)
            stat_for_table[method]["Top 1"][model_name] = deepcopy(top1)
            if is_top5:
                stat_for_table[method]["Top 5"][model_name] = deepcopy(top5)

            for res in [probabs, top1, correlation_p, correlation_s, topk_intersection]:
                for val in values:
                    res[val] = (np.mean(res[val], axis=0), sem(np.vstack(res[val]), axis=0))
            if is_top5:
                for val in values:
                    top5[val] = (np.mean(top5[val], axis=0), sem(np.vstack(top5[val]), axis=0))
            dict_acc[model_name] = {"Top-1": [top1[val][0] for val in values]}
            if is_top5:
                dict_acc[model_name]["Top-5"] = [top5[val][0] for val in values]

            prob[model_name] = ([probabs[val][0] for val in values],
                                [probabs[val][1] for val in values])
            stat[model_name]["Correlation Pearson"][method] = \
                ([correlation_p[val][0] for val in values],
                 [correlation_p[val][1] for val in values])
            stat[model_name]["Correlation Spearman"][method] = \
                ([correlation_s[val][0] for val in values],
                 [correlation_s[val][1] for val in values])
            stat[model_name]["Topk intersection"][method] = ([topk_intersection[val][0]
                                                              for val in values],
                                                             [topk_intersection[val][1]
                                                              for val in values])
    visualise_acc(dict_acc, values, aug, orig_cwd)
    visualise_probab(prob, values, aug, orig_cwd)
    visualise_expl(stat, values, aug, orig_cwd)
    save_stats(stat_for_table, values, models_to_process, aug, orig_cwd)
    _ = fig_to_table(stat_for_table, values, models_to_process, stat_for_table.keys(), N,
                     probability_drop)
    return


@hydra.main(config_path="../config", config_name="default.yaml")
def process_results(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    orig_cwd = hydra.utils.get_original_cwd()
    for aug, (min_value, max_value) in cfg.augmentations.items():
        results = {}
        print(aug)
        for model in cfg.models_to_process:
            res = []
            for method in cfg.expl_methods:
                file_name = orig_cwd + '/outputs/' + model + '_' \
                            + "ImageNet" + '_' + aug + '_' + method + '.csv'
                res.extend(read_results(file_name))
            results[model] = res
        results_by_img_id = sort_by_img_id(results, aug, only_correct=cfg.only_correct)
        aggregate_and_visualise(results_by_img_id, aug, min_value, max_value,
                                orig_cwd, cfg.N, cfg.models_to_process, cfg.probab_drop)


if __name__ == "__main__":
    process_results()
