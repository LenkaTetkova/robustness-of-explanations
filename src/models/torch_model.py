import torch
import torchvision
from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Model_ImageNet(nn.Module):
    def __init__(
            self, name, checkpoint=None, load_pretrained=False, use_model_ema=True,
    ) -> None:
        super().__init__()
        model_command = f'self.model = torchvision.models.{name}()'
        exec(model_command)
        if load_pretrained:
            self.model = load_checkpoint(checkpoint,
                                         self.model,
                                         use_model_ema=use_model_ema)

    def forward(self, data) -> torch.Tensor:
        if data.ndim != 4:
            raise ValueError(
                "Expected input is not a 4D tensor," f"instead it is a {data.ndim}D tensor."
            )
        return self.model(data)


def load_checkpoint(filepath, model, use_model_ema=False):
    print(f"Loading checkpoint from {filepath}.")
    checkpoint = torch.load(filepath, map_location="cpu")
    if use_model_ema:
        try:
            model.load_state_dict(checkpoint["model_ema"])
        except:
            state_dict = checkpoint["model_ema"]
            new_state_dict = {}
            for key, value in state_dict.items():
                if key[:7] == "module.":
                    new_key = key[7:]
                    new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict)
    else:
        if 'model_state_dict' in checkpoint.keys():
            key = 'model_state_dict'
        else:
            key = 'model'
        try:
            model.load_state_dict(checkpoint[key], strict=True)
        except:
            state_dict = {}
            for key, value in checkpoint[key].items():
                new_key = key[7:]
                state_dict[new_key] = value
            model.load_state_dict(state_dict, strict=True)

    if 'config' in checkpoint.keys():
        config = checkpoint['config']
    else:
        config = checkpoint['args']
    print(f"Model pretrained with config {config}.")
    return model


def save_checkpoint(model, filepath, cfg):
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg,
    }, filepath)


def load_model(model_name, pretrained, checkpoint_path,
               use_model_ema=False):
    layer = None
    model = Model_ImageNet(model_name,
                           checkpoint=checkpoint_path,
                           load_pretrained=pretrained,
                           use_model_ema=use_model_ema,
                           )
    if "resnet" in model_name:
        model.model = change_relu_resnet(model.model)
        layer = model.model.layer4
    elif "vgg" in model_name or model_name == "alexnet":
        model.model = change_relu_feat_cls(model.model)
        layer = model.model.features
    elif model_name == "mobilenet_v3_large":
        model.model = change_relu_mobilenet(model.model)
        layer = model.model.features
    elif model_name == "efficientnet_v2_m":
        model.model = change_silu_efficientnet(model.model)
        layer = model.model.features
    if not pretrained:
        model.model.apply(init_weights)
    return model, layer


def get_post_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    post_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])
    return post_transform


def change_relu_resnet(model):
    model.relu = torch.nn.ReLU(inplace=False)
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for key in layer._modules.keys():
            layer._modules[key].relu = torch.nn.ReLU(inplace=False)
    return model


def change_relu_mobilenet(model):
    for i in range(1, len(model.features._modules) - 1):
        for mod in model.features._modules[str(i)]._modules:
            for mod1 in model.features._modules[str(i)]._modules[mod]:
                for mod2 in mod1._modules:
                    if isinstance(mod1._modules[mod2], torch.nn.ReLU):
                        mod1._modules[mod2] = torch.nn.ReLU(inplace=False)
    return model


def change_relu_feat_cls(model):
    for i in range(len(model.classifier._modules)):
        if isinstance(model.classifier._modules[str(i)], torch.nn.ReLU):
            model.classifier._modules[str(i)] = torch.nn.ReLU(inplace=False)
    for i in range(len(model.features._modules)):
        if isinstance(model.features._modules[str(i)], torch.nn.ReLU):
            model.features._modules[str(i)] = torch.nn.ReLU(inplace=False)
    return model


def change_silu_efficientnet(model):
    for i in range(len(model.features._modules)):
        for j in range(len(model.features._modules[str(i)]._modules)):
            if isinstance(model.features._modules[str(i)]._modules[str(j)], torch.nn.SiLU):
                model.features._modules[str(i)]._modules[str(j)] = torch.nn.SiLU(inplace=False)
            elif 'block' in model.features._modules[str(i)]._modules[str(j)]._modules.keys():
                for k in range(
                    len(
                        model.features._modules[str(i)]._modules[str(j)]._modules['block']._modules
                    )
                ):
                    for m in range(
                        len(model.features._modules[str(i)]._modules[str(j)].
                            _modules['block']._modules[str(k)]._modules)
                    ):
                        if (str(m) in model.features._modules[str(i)]._modules[str(j)].
                                _modules['block']._modules[str(k)]._modules.keys() and
                                isinstance(model.features._modules[str(i)]._modules[str(j)].
                                           _modules['block']._modules[str(k)]._modules[str(m)],
                                           torch.nn.SiLU)):
                            model.features._modules[str(i)]._modules[str(j)]._modules['block'].\
                                _modules[str(k)]._modules[str(m)] = torch.nn.SiLU(inplace=False)
                        elif ("activation" in model.features._modules[str(i)]._modules[str(j)]
                                ._modules['block']._modules[str(k)]._modules.keys() and
                              isinstance(model.features._modules[str(i)]._modules[str(j)].
                                         _modules['block']._modules[str(k)].
                                         _modules['activation'],
                                         torch.nn.SiLU)):
                            model.features._modules[str(i)]._modules[str(j)].\
                                _modules['block']._modules[str(k)]._modules['activation']\
                                = torch.nn.SiLU(inplace=False)
    model.classifier._modules['0'] = torch.nn.Dropout(p=0.3, inplace=False)
    torch.autograd.set_grad_enabled(True)
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.features.parameters():
        param.requires_grad = True
    return model
