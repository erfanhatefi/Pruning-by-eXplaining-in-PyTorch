import yaml
import click
import torch
import tqdm.auto
import numpy as np
import wandb

from models import ModelLoader
from metrics import compute_accuracy
from datasets import ImageNet, ImageNetSubset, get_sample_indices_for_class


from utils import (
    initialize_random_seed,
    initialize_wandb_logger,
)

from pxp import (
    ModelLayerUtils,
    get_cnn_composite,
    get_vit_composite,
)


from pxp import GlobalPruningOperations
from pxp import ComponentAttibution


# generate some input reciever using click
@click.command()
@click.option("--configs_path", type=str)
@click.option("--output_path", type=str)
@click.option("--checkpoint_path", type=str)
@click.option("--dataset_path", type=str)
def start(
    configs_path,
    output_path,
    checkpoint_path,
    dataset_path,
):
    # Parse the configs file
    with open(configs_path, "r") as stream:
        configs = yaml.safe_load(stream)

    configs["output_path"] = output_path
    configs["checkpoint_path"] = checkpoint_path
    configs["dataset_path"] = dataset_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initialize_random_seed(configs["random_seed"])
    results_file_name = (
        f"accuracy_{configs['model_architecture']}_{configs['mode']}_{configs['subsequent_layer_pruning']}_sort{configs['abs_sort']}_leastvalue{configs['least_relevant_first']}_{configs['random_seed']}"
        if configs["mode"] == "Relevance"
        else f"accuracy_{configs['mode']}_{configs['subsequent_layer_pruning']}_sort{configs['abs_sort']}_leastvalue{configs['least_relevant_first']}_{configs['random_seed']}"
    )

    configs["results_file_name"] = results_file_name
    suggested_composite = {
        "low_level_hidden_layer_rule": configs["low_level_hidden_layer_rule"],
        "mid_level_hidden_layer_rule": configs["mid_level_hidden_layer_rule"],
        "high_level_hidden_layer_rule": configs["high_level_hidden_layer_rule"],
        "fully_connected_layers_rule": configs["fully_connected_layers_rule"],
        "softmax_rule": configs["softmax_rule"],
    }

    """
    Initialize WANDB run
    """
    if configs["wandb"]:
        initialize_wandb_logger(**configs)

    """
    Load the dataset
    """
    imagenet = ImageNet(
        configs["dataset_path"], random_seed=configs["random_seed"], num_workers=8
    )
    train_set = imagenet.get_train_set()
    val_set = imagenet.get_valid_set()

    list_random_classes = train_set.random_classes(
        configs["domain_restriction_classes"]
    )
    print(f"Random Classes chosen: {list_random_classes}")
    # For speeding up, we can use the pre-computed 10
    # randomly-chosen reference samples for attributing the model
    if configs["domain_restriction_classes"] == 1000:
        pruning_indices = torch.load(
            "datasets/imagenet_pruning_indices_1000_classes.pt"
        )
    else:
        pruning_indices = get_sample_indices_for_class(
            train_set,
            list_random_classes,
            configs["reference_samples_per_class"],
            "cuda",
        )

    validation_indices = get_sample_indices_for_class(
        val_set, list_random_classes, "all"
    )

    custom_pruning_set = ImageNetSubset(train_set, pruning_indices)
    custom_validation_set = ImageNetSubset(val_set, validation_indices)

    custom_pruning_dataloader = torch.utils.data.DataLoader(
        custom_pruning_set,
        batch_size=configs["pruning_dataloader_batchsize"],
        shuffle=True,
        num_workers=8,
    )
    custom_validation_dataloader = torch.utils.data.DataLoader(
        custom_validation_set,
        batch_size=configs["validation_dataloader_batchsize"],
        shuffle=True,
        num_workers=8,
    )

    del custom_pruning_set
    del custom_validation_set
    del validation_indices
    del pruning_indices
    del imagenet
    del train_set
    del val_set

    """
    Load the model and applying the Composites/Canonizers
    """
    if configs["model_architecture"] == "vit_b_16":
        composite = get_vit_composite(
            configs["model_architecture"], suggested_composite
        )
    else:
        composite - get_cnn_composite(
            configs["model_architecture"], suggested_composite
        )

    model = ModelLoader.get_basic_model(
        configs["model_architecture"], configs["checkpoint_path"], device
    )

    layer_types = {
        "Softmax": torch.nn.Softmax,
        "Linear": torch.nn.Linear,
        "Conv2d": torch.nn.Conv2d,
    }

    # modify model's last linear layer for domain
    # restriction (e.g., 3 classes classification)
    if len(list_random_classes) != 1000:
        ModelLayerUtils.mask_last_layer_domain_restriction(model, list_random_classes)

    """
    Setting up extra configurations for the pruning
    """
    pruning_rates = configs["pruning_rates"]

    component_attributor = ComponentAttibution(
        "Relevance",
        "ViT" if configs["model_architecture"] == "vit_b_16" else "CNN",
        layer_types[configs["pruning_layer_type"]],
    )

    components_relevances = component_attributor.attribute(
        model,
        custom_pruning_dataloader,
        composite,
        abs_flag=True,
        device=device,
    )

    acc_top1 = compute_accuracy(
        model,
        custom_validation_dataloader,
        device,
    )
    print(f"Initial accuracy: top1={acc_top1}")

    """
    Experiment's main loop
    """
    layer_names = component_attributor.layer_names
    pruner = GlobalPruningOperations(
        layer_types[configs["pruning_layer_type"]],
        layer_names,
    )
    top1_acc_list = []
    progress_bar = tqdm.tqdm(total=len(pruning_rates))
    for pruning_rate in pruning_rates:
        progress_bar.set_description(f"Processing {int((pruning_rate)*100)}% Pruning")
        # skip pruning if compression rate is 0.00 as we
        # have computed few lines above, otherwise prune
        if pruning_rate != 0.0:
            # prune the model based on the
            # pre-computed attibution flow
            # (relevance values)
            global_pruning_mask = pruner.generate_global_pruning_mask(
                model,
                components_relevances,
                pruning_rate,
                subsequent_layer_pruning=configs["subsequent_layer_pruning"],
                least_relevant_first=configs["least_relevant_first"],
                device=device,
            )
            # Our pruning gets applied by masking the
            # activation of layers via forward hooks.
            # Therefore hooks are returned for later
            # removal
            hook_handles = pruner.fit_pruning_mask(
                model,
                global_pruning_mask,
            )

        progress_bar.set_description(
            f"Computing accuracy for model prunned with {int((pruning_rate)*100)}%"
        )
        acc_top1 = compute_accuracy(
            model,
            custom_validation_dataloader,
            device,
        )
        # Remove/Deactivate hooks (except
        # when the pruning rate is 0.00)
        if pruning_rate != 0.0:
            if layer_types[configs["pruning_layer_type"]] == torch.nn.Softmax:
                for hook in hook_handles:
                    hook.remove()
        top1_acc_list.append(acc_top1)
        print(f"Accuracy-Flow list: {top1_acc_list}")

        """
        Logging the results on WandB
        """
        if configs["wandb"]:
            wandb.log({"acc_top1": acc_top1, "pruning_rate": pruning_rate})
            print(f"Logged the results of {pruning_rate}% Pruning Rate to wandb!")
        progress_bar.update(1)

    # empty up the GPU memory and CUDA cache, model and dataset
    progress_bar.close()
    del pruner
    torch.cuda.empty_cache()

    top1_auc = compute_auc(top1_acc_list, pruning_rates)
    if configs["wandb"]:
        wandb.log({"top1_auc": top1_auc})
        print(f"Logged the AUC of the Top1 Accuracy to wandb!")

    print(f"Top1 AUC: {top1_auc}")


def compute_auc(top1_acc_list, pruning_rates):
    """
    Compute the Area Under the Curve (AUC) for the accuracy over the pruning rates
    """
    top1_auc = np.trapz(top1_acc_list, pruning_rates)
    print(f"Top1 AUC: {top1_auc}")
    return top1_auc


if __name__ == "__main__":
    start()
