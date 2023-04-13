import time

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from captum.attr import (LRP, Deconvolution, DeepLift, FeatureAblation,
                         GradientShap, GuidedBackprop, GuidedGradCam,
                         InputXGradient, IntegratedGradients, KernelShap,
                         Occlusion, Saliency, ShapleyValueSampling)
from scipy.stats import sem

from src.explanations.compare_explanations import pixel_flipping_AOC
from src.explanations.explain_zennit import explain_one_zennit
from src.explanations.utils import accuracy_and_prob


def explain_captum(model,
                   images,
                   labels,
                   explainer,
                   device,
                   masks,
                   folder,
                   top5=True,
                   model_layer=None,
                   model_name="resnet50",
                   n_classes=1000,
                   ):
    if "zennit" not in explainer and explainer != "random":
        expl_func = get_explainer(explainer, model, model_layer)
    explanations = []
    PF_scores = []
    PF_scores_all = []
    end = time.time()
    probs = []
    acc = {
        "top-1": []
    }
    if max(labels) < 5:
        top5 = False
    if top5:
        acc["top-5"] = []
    topk = (1, 5) if top5 else (1,)

    for idx, (imgs, target, masks_) in enumerate(zip(images, labels, masks)):
        att = []
        top = {
            "top-1": []
        }
        if top5:
            top["top-5"] = []
        prob_ = []
        for split, m_split in zip(imgs, masks_):
            model = model.to(device)
            target = torch.as_tensor(target)

            split = torch.unsqueeze(split, 0)
            split = split.to(device)

            # compute output
            output = model(split)

            output.data = output.data.to('cpu')
            predictions = np.argmax(output.data, axis=-1)
            # measure accuracy and probabilities
            prec, prob = accuracy_and_prob(output.data, target, topk=topk)
            for key in prec.keys():
                top[key].append(prec[key])
            prob_.append(prob)
            target = target.to(device)

            # Create explanations
            if "zennit" in explainer:
                attributions = explain_one_zennit(explainer, model, split, target, device,
                                                  model_type=model_name, classes=n_classes)
            else:
                attributions = explain_one(explainer, expl_func, split, target)
            att.extend(attributions)
            split = split.to('cpu').detach().numpy()
            target = target.to('cpu').detach().numpy()

        explanations.append(att)
        model = model.to('cpu')
        attr = att[0].to('cpu').detach().numpy()
        attr = np.sum(attr, axis=0, keepdims=True)
        if explainer != "zennit-Occlusion":
            try:
                PF_sc, sc_all = pixel_flipping_AOC(np.expand_dims(imgs[0].numpy(), 0),
                                                   np.array([target]),
                                                   np.expand_dims(attr, 0),
                                                   model,
                                                   512,
                                                   )
                PF_scores.append(PF_sc)
                PF_scores_all.append(sc_all)
            except AssertionError as E:
                print(f"Index {idx}: {E}")

        for key in top.keys():
            acc[key].append(top[key])
        probs.append(prob_)

    # measure elapsed time
    print(time.strftime("%H:%M:%S +0000", time.gmtime(time.time() - end)))
    if top5:
        print(' * Prec@1 {top1:.3f} Prec@5 {top5:.3f}'
              .format(top1=np.mean(acc["top-1"]), top5=np.mean(acc["top-5"])))
    else:
        print(' * Prec@1 {top1:.3f}'
              .format(top1=np.mean(acc["top-1"])))
    if explainer != "zennit-Occlusion":
        PF_mean = np.mean(PF_scores)
        PF_sem = sem(PF_scores)
        print(f'{explainer}: Pixel flipping: {PF_mean}+-{PF_sem}')
    return {explainer: explanations}, {explainer: acc}, {explainer: probs}, PF_scores_all


def get_explainer(name, model, layer=None):
    explainer_dict = {
        "IntegratedGradients": IntegratedGradients(model),
        "Gradients": Saliency(model),
        "InputGradients": InputXGradient(model),
        "GuidedBackprop": GuidedBackprop(model),
        "Deconvolution": Deconvolution(model),
        "LRP": LRP(model),
        "ShapleyValueSampling": ShapleyValueSampling(model),
        "DeepLift": DeepLift(model, multiply_by_inputs=True),
        "KernelShap": KernelShap(model),
        "Occlusion": Occlusion(model),
        "FeatureAblation": FeatureAblation(model),
        "GradientShap": GradientShap(model, multiply_by_inputs=True),
    }
    if layer is not None:
        explainer_dict["GuidedGradCam"] = GuidedGradCam(model, layer)
    return explainer_dict[name]


def explain_one(explainer, expl_func, split, targets):
    if explainer == 'IntegratedGradients':
        attributions = expl_func.attribute(split, 0, target=targets,
                                           return_convergence_delta=False)
    elif explainer == 'Occlusion':
        attributions = expl_func.attribute(split, target=targets,
                                           sliding_window_shapes=(3, 3, 3))
    elif explainer == "Gradients":
        attributions = expl_func.attribute(split, target=targets, abs=False)
    else:
        attributions = expl_func.attribute(split, target=targets)

    attributions = attributions.to('cpu')
    return attributions
