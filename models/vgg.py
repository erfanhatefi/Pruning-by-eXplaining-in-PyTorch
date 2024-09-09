import torch
import torchvision

"""
    This module, only returns the architecture of the 
    resnet with predefined(initialized by default) wights
"""


def load_vgg19_bn(out_features, pretrained=True):
    """
    Loading vgg19_bn from torchvision models

    Args:
        out_features (int): number of neurons at the last layer as the model's output
        pretrained (bool, optional): If True, load the model with pretrained weights. Defaults to True.

    Returns:
        model (torch.nn.module): the model
    """
    if pretrained:
        from torchvision.models.vgg import VGG19_BN_Weights

        model = torchvision.models.vgg19_bn(weights=VGG19_BN_Weights)
    else:
        model = torchvision.models.vgg19_bn()

    if out_features != 1000:
        in_features = model._modules["classifier"][-1].in_features
        model._modules["classifier"] = model._modules["classifier"][:-1]
        model._modules["classifier"].add_module(
            "6", torch.nn.Linear(in_features, out_features)
        )

    return model


def load_vgg19(out_features, pretrained=True):
    """
    Loading vgg19 from torchvision models

    Args:
        out_features (int): number of neurons at the last layer as the model's output
        pretrained (bool, optional): If True, load the model with pretrained weights. Defaults to True.

    Returns:
        model (torch.nn.module): the model
    """
    if pretrained:
        from torchvision.models.vgg import VGG19_Weights

        model = torchvision.models.vgg19(weights=VGG19_Weights)
    else:
        model = torchvision.models.vgg19()

    if out_features != 1000:
        in_features = model._modules["classifier"][-1].in_features
        model._modules["classifier"] = model._modules["classifier"][:-1]
        model._modules["classifier"].add_module(
            "6", torch.nn.Linear(in_features, out_features)
        )

    return model


def load_vgg16_bn(out_features, pretrained=True):
    """
    Loading vgg16_bn from torchvision models

    Args:
        out_features (int): number of neurons at the last layer as the model's output
        pretrained (bool, optional): If True, load the model with pretrained weights. Defaults to True.

    Returns:
        model (torch.nn.module): the model
    """
    if pretrained:
        from torchvision.models.vgg import VGG16_BN_Weights

        model = torchvision.models.vgg16_bn(weights=VGG16_BN_Weights)
    else:
        model = torchvision.models.vgg16_bn()

    if out_features != 1000:
        in_features = model._modules["classifier"][-1].in_features
        model._modules["classifier"] = model._modules["classifier"][:-1]
        model._modules["classifier"].add_module(
            "6", torch.nn.Linear(in_features, out_features)
        )

    return model


def load_vgg16(out_features, pretrained=True):
    """
    Loading vgg16 from torchvision models

    Args:
        out_features (int): number of neurons at the last layer as the model's output
        pretrained (bool, optional): If True, load the model with pretrained weights. Defaults to True.

    Returns:
        model (torch.nn.module): the model
    """
    if pretrained:
        from torchvision.models.vgg import VGG16_Weights

        model = torchvision.models.vgg16(weights=VGG16_Weights)
    else:
        model = torchvision.models.vgg16()

    if out_features != 1000:
        in_features = model._modules["classifier"][-1].in_features
        model._modules["classifier"] = model._modules["classifier"][:-1]
        model._modules["classifier"].add_module(
            "6", torch.nn.Linear(in_features, out_features)
        )

    return model


def load_vgg13_bn(out_features, pretrained=True):
    """
    Loading vgg16_bn from torchvision models

    Args:
        out_features (int): number of neurons at the last layer as the model's output
        pretrained (bool, optional): If True, load the model with pretrained weights. Defaults to True.

    Returns:
        model (torch.nn.module): the model
    """
    if pretrained:
        from torchvision.models.vgg import VGG13_BN_Weights

        model = torchvision.models.vgg13_bn(weights=VGG13_BN_Weights)
    else:
        model = torchvision.models.vgg13_bn()

    if out_features != 1000:
        in_features = model._modules["classifier"][-1].in_features
        model._modules["classifier"] = model._modules["classifier"][:-1]
        model._modules["classifier"].add_module(
            "6", torch.nn.Linear(in_features, out_features)
        )

    return model


def load_vgg13(out_features, pretrained=True):
    """
    Loading vgg13 from torchvision models

    Args:
        out_features (int): number of neurons at the last layer as the model's output
        pretrained (bool, optional): If True, load the model with pretrained weights. Defaults to True.

    Returns:
        model (torch.nn.module): the model
    """
    if pretrained:
        from torchvision.models.vgg import VGG13_Weights

        model = torchvision.models.vgg13(weights=VGG13_Weights)
    else:
        model = torchvision.models.vgg13()

    if out_features != 1000:
        in_features = model._modules["classifier"][-1].in_features
        model._modules["classifier"] = model._modules["classifier"][:-1]
        model._modules["classifier"].add_module(
            "6", torch.nn.Linear(in_features, out_features)
        )

    return model
