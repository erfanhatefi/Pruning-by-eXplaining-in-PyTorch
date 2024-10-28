from torchvision.models.vision_transformer import (
    vit_b_16,
    ViT_B_16_Weights,
)
import torch


def load_vit_b_16(checkpoint_path):
    """
    Loading ViT-B-16 from torchvision models

    Args:
        checkpoint_path (str): checkpoint-path of the model if given

    Returns:
        model (torch.nn.module): the model
    """
    weights = None
    if checkpoint_path is None:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
    else:
        weights = torch.load(checkpoint_path)
    model = vit_b_16(weights=weights)
    del weights
    model.eval()

    return model
