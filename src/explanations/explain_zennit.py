;import torch
from torchvision.transforms import Normalize
from zennit.attribution import Gradient, IntegratedGradients, Occlusion
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import (EpsilonAlpha2Beta1Flat, EpsilonGammaBox,
                               EpsilonPlusFlat)
from zennit.rules import Pass
from zennit.torchvision import ResNetCanonizer, VGGCanonizer

from src.explanations.lrp_canonizers import (EfficientNetCanonizer,
                                             SqueezeExcitation)


def explain_one_zennit(explainer, model, split, targets, device, classes=1000,
                       model_type="resnet"):
    layer_map = []
    if "resnet" in model_type:
        canonizer = ResNetCanonizer()
    elif "vgg" in model_type:
        canonizer = VGGCanonizer()
    elif "efficientnet" in model_type:
        canonizer = EfficientNetCanonizer()
        layer_map = [(SqueezeExcitation, Pass())]
    else:
        canonizer = SequentialMergeBatchNorm()
    if explainer == 'zennit-Gradients':
        attributor = Gradient(model=model)
    elif explainer == 'zennit-IntegratedGradients':
        attributor = IntegratedGradients(model=model)
    elif explainer == 'zennit-Occlusion':
        attributor = Occlusion(model=model)
    elif explainer == 'zennit-EpsilonPlusFlat':
        composite = EpsilonPlusFlat(canonizers=[canonizer], layer_map=layer_map)
        attributor = Gradient(model=model, composite=composite)
    elif explainer == 'zennit-EpsilonAlpha2Beta1Flat':
        composite = EpsilonAlpha2Beta1Flat(canonizers=[canonizer], layer_map=layer_map)
        attributor = Gradient(model=model, composite=composite)
    elif explainer == 'zennit-EpsilonGammaBox':
        transform_norm = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        low, high = transform_norm(torch.tensor([[[[[0.]]] * 3], [[[[1.]]] * 3]]))
        composite = EpsilonGammaBox(low=low, high=high, canonizers=[canonizer],
                                    layer_map=layer_map)
        attributor = Gradient(model=model, composite=composite)
    targ = torch.zeros(1, classes)
    targ[0, targets.item()] = 1
    targ = targ.to(device)
    output, attributions = attributor(split, targ)
    targ = targ.to('cpu')

    attributions = attributions.to('cpu')
    return attributions
