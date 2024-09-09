import torch
import torchvision

"""
    This module, only returns the architecture of the 
    resnet with predefined(initialized by default) wights
"""


def load_resnet18(out_features, pretrained=True):
    """
    Loading resnet18 from torchvision models

    Args:
        out_features (int): number of neurons at the last layer as the model's output
        pretrained (bool, optional): If True, load the model with pretrained weights. Defaults to True.

    Returns:
        model (torch.nn.module): the model
    """
    if pretrained:
        print("Load pretrained model")
        from torchvision.models.resnet import ResNet18_Weights

        model = torchvision.models.resnet18(weights=ResNet18_Weights)
    else:
        model = torchvision.models.resnet18()

    if out_features != 1000:
        in_features = model._modules["fc"].in_features
        model._modules["fc"] = torch.nn.Linear(in_features, out_features)

    return model


def load_resnet34(out_features, pretrained=True):
    """
    Loading resnet34 from torchvision models

    Args:
        out_features (int): number of neurons at the last layer as the model's output
        pretrained (bool, optional): If True, load the model with pretrained weights. Defaults to True.

    Returns:
        model (torch.nn.module): the model
    """
    if pretrained:
        print("Load pretrained model")
        from torchvision.models.resnet import ResNet34_Weights

        model = torchvision.models.resnet34(weights=ResNet34_Weights)
    else:
        model = torchvision.models.resnet34()

    if out_features != 1000:
        in_features = model._modules["fc"].in_features
        model._modules["fc"] = torch.nn.Linear(in_features, out_features)

    return model


def load_resnet50(out_features, pretrained=True):
    """
    Loading resnet50 from torchvision models

    Args:
        out_features (int): number of neurons at the last layer as the model's output
        pretrained (bool, optional): If True, load the model with pretrained weights. Defaults to True.

    Returns:
        model (torch.nn.module): the model
    """
    if pretrained:
        print("Load pretrained model")
        from torchvision.models.resnet import ResNet50_Weights

        model = torchvision.models.resnet50(weights=ResNet50_Weights)
    else:
        model = torchvision.models.resnet50()

    if out_features != 1000:
        in_features = model._modules["fc"].in_features
        model._modules["fc"] = torch.nn.Linear(in_features, out_features)

    return model


def load_resnet101(out_features, pretrained=True):
    """
    Loading resnet101 from torchvision models

    Args:
        out_features (int): number of neurons at the last layer as the model's output

    Returns:
        model (torch.nn.module): the model
        model_name (str): the name of the model's architecture
    """
    if pretrained:
        print("Load pretrained model")
        from torchvision.models.resnet import ResNet101_Weights

        model = torchvision.models.resnet101(weights=ResNet101_Weights)
    else:
        model = torchvision.models.resnet101()

    if out_features != 1000:
        in_features = model._modules["fc"].in_features
        model._modules["fc"] = torch.nn.Linear(in_features, out_features)

    return model


def load_resnet152(out_features, pretrained=True):
    """
    Loading resnet152 from torchvision models

    Args:
        out_features (int): number of neurons at the last layer as the model's output
        pretrained (bool, optional): If True, load the model with pretrained weights. Defaults to True.

    Returns:
        model (torch.nn.module): the model
    """
    if pretrained:
        print("Load pretrained model")
        from torchvision.models.resnet import ResNet152_Weights

        model = torchvision.models.resnet152(weights=ResNet152_Weights)
    else:
        model = torchvision.models.resnet152()

    if out_features != 1000:
        in_features = model._modules["fc"].in_features
        model._modules["fc"] = torch.nn.Linear(in_features, out_features)

    return model
