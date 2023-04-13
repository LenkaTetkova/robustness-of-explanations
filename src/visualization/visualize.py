from typing import List

import matplotlib.pyplot as plt
import numpy as np


def visualise_expl(stat, values, name_aug, folder, save=True, std_avg=None):
    if std_avg:
        values = values / std_avg
        x_label = "std of noise / average std of original images"
    else:
        x_label = "augmentation values"
    new_stat = {}
    for name_model, val_model in stat.items():
        for name_metr, val_metr in val_model.items():
            new_stat[name_metr] = {}
    for name_model, val_model in stat.items():
        for name_metr, val_metr in val_model.items():
            new_stat[name_metr][name_model] = val_metr
    for method, results in new_stat.items():
        fig = plt.figure()
        for name_model, model_results in results.items():
            for name, vals in model_results.items():
                mean, std = vals
                mean = np.array(mean)
                std = np.array(std).flatten()
                plt.plot(values, mean, label=f"{name_model}: {name}")
                plt.fill_between(values,
                                 mean - std,
                                 mean + std,
                                 color='gray',
                                 alpha=0.2,
                                 )
        plt.ylim([0, 1])
        plt.title(name_aug + ' - ' + method)
        legend = plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.xlabel(x_label)
        plt.ylabel(method)
        if save:
            plt.savefig(folder
                        + '/reports/figures/'
                        + name_aug
                        + '-'
                        + method
                        + '.png',
                        bbox_extra_artists=(legend,),
                        bbox_inches='tight'
                        )
        plt.show()
        plt.close(fig)


def visualise_probab(prob, values, name_aug, folder, save=True, std_avg=None):
    fig = plt.figure()
    if std_avg:
        values = values / std_avg
        x_label = "std of noise / average std of original images"
    else:
        x_label = "augmentation values"
    for method, results in prob.items():
        mean, std = results
        mean = np.array(mean)
        std = np.array(std).flatten()
        plt.plot(values, mean, label=method)
        plt.fill_between(values,
                         mean - std,
                         mean + std,
                         color='gray',
                         alpha=0.2,
                         )
    plt.ylim([0, 1])
    keys = prob.keys()
    fig_name = "-".join(keys)
    plt.title(f"{name_aug}: {fig_name}")
    legend = plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.xlabel(x_label)
    plt.ylabel("probability")
    if save:
        plt.savefig(folder
                    + '/reports/figures/'
                    + "Probabilities"
                    + '-'
                    + name_aug
                    + '-'
                    + fig_name
                    + '.png',
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight'
                    )
    plt.show()
    plt.close(fig)


def visualise_acc(accuracy, values: List[float], name_aug, folder, save=True, std_avg=None):
    if std_avg:
        values = values / std_avg
        x_label = "std of noise / average std of original images"
    else:
        x_label = "augmentation values"
    fig = plt.figure()
    for model_name, model_acc in accuracy.items():
        for key, acc in model_acc.items():
            plt.plot(values, acc, label=f"{model_name}: {key}")
    plt.ylim([0, 1])
    plt.title(name_aug + ' - ' + 'Accuracy')
    plt.xlabel(x_label)
    plt.ylabel("accuracy")
    plt.legend()
    if save:
        plt.savefig(folder + '/reports/figures/' + name_aug + '-' + 'acc' + '.png')
    plt.show()
    plt.close(fig)
