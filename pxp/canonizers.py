from zennit.torchvision import VGGCanonizer, ResNetCanonizer
from pxp.LiT_utils.vit_canonizer import (
    VitTorchvisionSumCanonizer,
    ReplaceAttentionNorm,
    ReplaceAttention,
)


def get_cnn_canonizer(model_architecture):
    """
    Get the canonizer for the model architecture (CNN-based)

    Args:
        model_architecture (str): model architecture

    Returns:
        zennit.canonizer: canonizer for the model architecture
    """
    canonizer_wrapper = {
        "vgg16_bn": VGGCanonizer,
        "resnet18": ResNetCanonizer,
        "resnet50": ResNetCanonizer,
    }
    if model_architecture not in ["Linear", "vgg16"]:
        return [canonizer_wrapper[model_architecture]()]
    else:
        return None


def get_vit_canonizer(canonizations):

    canonizer_wrapper = {
        "VitTorchvisionSumCanonizer": VitTorchvisionSumCanonizer,
        "ReplaceAttentionNorm": ReplaceAttentionNorm,
        "ReplaceAttention": ReplaceAttention,
    }
    return [canonizer_wrapper[canonization]() for canonization in canonizations]
