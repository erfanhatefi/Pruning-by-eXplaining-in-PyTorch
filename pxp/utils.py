from typing import List
import torch


def one_hot_max(output, targets):
    """
    Get the one-hot encoded max at the original indices in dim=1

    Args:
        output (torch.tensor): the output of the model
        targets (torch.tensor): the targets of the model

    Returns:
        torch.tensor: the one-hot encoded matrix multiplied by
                        the targets logit at the original indices in dim=1
    """
    values = output[torch.arange(output.shape[0]).unsqueeze(-1), targets.unsqueeze(-1)]
    eye_matrix = torch.eye(output.shape[-1]).to(output.device)
    return values * eye_matrix[targets]


def one_hot(output, targets):
    """
    Get the one-hot encoded at the original indices in dim=1

    Args:
        output (torch.tensor): the output of the model
        targets (torch.tensor): the targets of the model

    Returns:
        torch.tensor: the one-hot encoded matrix multiplied by
                        the targets logit at the original indices in dim=1
    """
    eye_matrix = torch.eye(output.shape[-1]).to(output.device)
    return eye_matrix[targets]


class ModelLayerUtils:

    @staticmethod
    def get_layer_names(model: torch.nn.Module, types: List):
        """
        GOT FROM ZENNIT-CRP'S REPO
        Retrieves the layer names of all layers that belong to a torch.nn.Module type defined
        in 'types'.

        Parameters
        ----------
        model: torch.nn.Module
        types: list of torch.nn.Module
            Layer types i.e. torch.nn.Conv2D

        Returns
        -------
        layer_names: list of strings
        """

        layer_names = []

        for name, layer in model.named_modules():
            for layer_definition in types:
                if isinstance(layer, layer_definition) or issubclass(
                    layer.__class__, layer_definition
                ):
                    if name not in layer_names:
                        layer_names.append(name)

        return layer_names

    @staticmethod
    def get_module_from_name(model, complete_layer_name):
        """
        Get the module from the name extracted via crp.helper.get_layer_names.
        Compared to the previous methods, this method can handle nested modules
        like the torch.nn.Sequential, and other modules that are not the basic.
        By basics we mean:
            torch.nn.Conv2d,
            torch.nn.BatchNorm2d,
            torch.nn.ReLU,
            torch.nn.Dropout,
            torch.nn.Linear,
            torch.nn.MaxPool2d,

        Args:
            model (torch.nn.Module): the model or the module
            complete_layer_name (str): the name of the layer
        """

        def nested_getattr(object, attribute):
            for attr in attribute.split("."):
                object = getattr(object, attr)
            return object

        return nested_getattr(model, complete_layer_name)

    @staticmethod
    def parse_layer(layer_name):
        """
        Given a layer name, parse it into the sub_layer_name and layer_index

        Args:
            layer_name (str): name of the layer based on the torchvision model

        Returns:
            Tuple: pair of sub_layer_name and layer_index
        """
        sub_layer_name, layer_index = layer_name.split(".")
        layer_index = int(layer_index)
        return sub_layer_name, layer_index

    @staticmethod
    def is_batchnorm2d_after_conv2d(model):
        basic_modules = [
            torch.nn.Conv2d,
            torch.nn.BatchNorm2d,
            torch.nn.ReLU,
            torch.nn.Dropout,
            torch.nn.Linear,
            torch.nn.MaxPool2d,
        ]

        def is_not_basic_module(module):
            return not isinstance(module, tuple(basic_modules))

        def is_conv2d_layer(module):
            if isinstance(module, torch.nn.Conv2d):
                return True
            else:
                return False

        def is_batchnorm_layer(module):
            return isinstance(module, torch.nn.BatchNorm2d)

        def check_order(module_list):
            for i in range(len(module_list) - 1):
                if not is_conv2d_layer(module_list[i]) or not is_batchnorm_layer(
                    module_list[i + 1]
                ):
                    return False
            return True

        # Iterate through the model's modules and their children
        for name, module in model.named_children():
            if is_conv2d_layer(module):
                # Check if Conv2d is followed by BatchNorm2d
                children = list(module.children())
                if not check_order(children):
                    return False

            elif is_not_basic_module(module):
                # If the module is an nn.Sequential, check the order of its children
                for seq_name, seq_module in module.named_children():
                    if is_not_basic_module(seq_module):
                        # Handle nested nn.Sequential layers recursively
                        if not ModelLayerUtils.is_batchnorm2d_after_conv2d(seq_module):
                            return False
                    elif is_conv2d_layer(seq_module):
                        # Check if Conv2d is followed by BatchNorm2d in the Sequential block
                        seq_children = list(seq_module.children())
                        if not check_order(seq_children):
                            return False

        return True

    @staticmethod
    def mask_last_layer_domain_restriction(model, list_classes):
        """
        Mask the last layer of the model with the given classes

        Args:
            model (torch.nn.module): model
            list_classes (list): list of given classes
        """
        last_layer_name = ModelLayerUtils.get_layer_names(model, [torch.nn.Linear])[-1]
        last_layer_module = ModelLayerUtils.get_module_from_name(model, last_layer_name)

        weight = last_layer_module.weight.data
        bias = last_layer_module.bias.data

        mask_weight = torch.zeros_like(weight)
        mask_weight[list_classes] = 1
        mask_bias = torch.zeros_like(bias)
        mask_bias[list_classes] = 1

        # replpace weights and biases with new values
        weight = weight * mask_weight
        bias = bias * mask_bias

        # bind new values to the model
        last_layer_module.weight.data = weight
        last_layer_module.bias.data = bias

    @staticmethod
    def disable_inplace_operations(model):
        for _, module in model.named_modules():
            if "inplace" in list(module.__dict__.keys()):
                module.inplace = False
