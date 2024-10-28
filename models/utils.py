from .resnet import *
from .vgg import *
from .vit import *

import torch


class ModelLoader:

    @staticmethod
    def load_model(architecture, num_classes=1000, **kwargs):
        """
        model loader

        Args:
            architecture (str): the name of the architecture
            norm_layer (str): the type of NormLayer you want
                            to use, aux(for AdvProp) and
                            None for the others
            num_gpus (int): the number of gpus

        Returns:
            model (torch.nn.module): the model
            model_name (str): the name of the model's architecture
        """
        model_dict = {
            "resnet18": {"loader": load_resnet18, "params": (num_classes, True)},
            "resnet34": {"loader": load_resnet34, "params": (num_classes, True)},
            "resnet50": {"loader": load_resnet50, "params": (num_classes, True)},
            "resnet101": {"loader": load_resnet101, "params": (num_classes, True)},
            "resnet152": {"loader": load_resnet152, "params": (num_classes, True)},
            "vgg19": {"loader": load_vgg19, "params": (num_classes, True)},
            "vgg19_bn": {"loader": load_vgg19_bn, "params": (num_classes, True)},
            "vgg16": {"loader": load_vgg16, "params": (num_classes, True)},
            "vgg16_bn": {"loader": load_vgg16_bn, "params": (num_classes, True)},
            "vgg13": {"loader": load_vgg13, "params": (num_classes, True)},
            "vgg13_bn": {"loader": load_vgg13_bn, "params": (num_classes, True)},
            "vit_b_16": {"loader": load_vit_b_16, "params": (None,)},
        }

        model = model_dict[architecture]["loader"](*model_dict[architecture]["params"])

        return model, architecture

    @staticmethod
    def get_basic_model(architecture, checkpoint_path, device):
        print(f"Arch:{architecture}")
        model, _ = ModelLoader.load_model(architecture, num_classes=1000)
        if checkpoint_path not in [None, "None"]:
            loaded_state_dictionary = ModelLoader.load_state_dictionary_from_checkpoint(
                checkpoint_path
            )
            model.load_state_dict(loaded_state_dictionary)
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def load_state_dictionary_from_checkpoint(checkpoint_path):
        """
        Loads the state dictionary from the checkpoint

        Args:
            checkpoint_path (str): the path to the checkpoint

        Returns:
            state_dict (dict): the state dictionary
        """
        loaded_checkpoint = torch.load(checkpoint_path)
        if "state_dict" in loaded_checkpoint.keys():
            state_dict = loaded_checkpoint["state_dict"]
        else:
            state_dict = loaded_checkpoint
        return state_dict
