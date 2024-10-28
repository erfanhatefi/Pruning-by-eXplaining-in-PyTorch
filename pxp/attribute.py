from collections import OrderedDict
import torch
from pxp.utils import one_hot_max, one_hot, ModelLayerUtils
from pxp.canonizers import get_vit_canonizer
from pxp.LiT.rules import (
    InputRefXGSoftmaxZPlusModuleVMAP,
    IxGSoftmaxBriefModule,
    BlockModule,
)


class LatentRelevanceAttributor:
    def __init__(self, layers_list_to_track) -> None:
        """
        Constructor

        Args:
            layers_list_to_track (list): list of types of layers to track
        """
        self.layers_list_to_track = layers_list_to_track
        self.latent_output = {}

    def lrp_pass(self, model, inputs, targets, composite, initial_relevance, device):
        """
        Compute the relevance using LRP

        Args:
            model (torch.nn.module): the model to be explained
            inputs (torch.tensor): inputs or the given images
            targets (torch.tensor): targets of the given images
            composite (): lrp composite
            device (): device to be used

        Returns:
            (torch.tensor): the computed heatmap using LRP
        """

        if initial_relevance == 1:
            initial_relevance_function = one_hot
        elif initial_relevance == "logit":
            initial_relevance_function = one_hot_max

        with torch.enable_grad():
            inputs.requires_grad = True
            # If composite has not been registered
            # in the model and is specified
            if composite != None:
                composite = composite
                with composite.context(model) as modeified_model:
                    relevance = self.compute_relevance(
                        modeified_model,
                        inputs,
                        targets,
                        initial_relevance_function,
                        device,
                    )
            # If the composite is already registered in the model
            else:
                # print("No composite specified")
                relevance = self.compute_relevance(
                    model, inputs, targets, initial_relevance_function, device
                )

        self.remove_hooks()
        self.parse_latent_relevances(model)

        return relevance

    def compute_relevance(
        self, model, inputs, targets, initial_relevance_function, device
    ):
        self.clear_latent_info()
        self.hook_handles = self.register_hooks(model)
        output = model(inputs)
        (relevance,) = torch.autograd.grad(
            outputs=output,
            inputs=inputs,
            grad_outputs=initial_relevance_function(output, targets).to(device),
            retain_graph=False,
            create_graph=False,
        )

        # detach the relevance to avoid memory leak
        return relevance.detach().cpu()

    def remove_hooks(self):
        """
        Remove the hooks
        """
        for handle in self.hook_handles:
            handle.remove()


class ZennitLatentRelevanceAttributor(LatentRelevanceAttributor):
    def __init__(self, layers_list_to_track) -> None:
        """
        Constructor

        Args:
            layers_list_to_track (list): list of types of layers to track
        """
        super().__init__(layers_list_to_track)
        self.latent_output = {}

    def parse_latent_relevances(self, model):
        """
        Extract the relevance values from the output tensors
        via .grad

        Args:
            model (torch.nn.module): model
        """
        self.latent_relevances = {}
        self.latent_activations = {}
        for layer_name in self.layers_list_to_track:
            for name, module in model.named_modules():
                if name == layer_name:
                    self.latent_relevances[name] = (
                        self.latent_output[name].grad.detach().cpu()
                    )
                    self.latent_output[name].detach().cpu()
                    self.latent_activations[name] = (
                        self.latent_output[name].detach().cpu()
                    )
                    # Sum the relevance over the spatial dimensions
                    # to acquire the component's relevance
                    if isinstance(module, torch.nn.Linear):
                        # check if there is extra dimension for the tokens
                        if len(self.latent_relevances[name].shape) == 3:
                            self.latent_relevances[name] = (
                                self.latent_relevances[name].detach().cpu()
                            )
                            self.latent_activations[name] = (
                                self.latent_activations[name].detach().cpu()
                            )
                    elif isinstance(module, torch.nn.Conv2d):
                        self.latent_relevances[name] = (
                            self.latent_relevances[name].sum(dim=(2, 3)).detach().cpu()
                        )
                        self.latent_activations[name] = (
                            self.latent_activations[name].sum(dim=(2, 3)).detach().cpu()
                        )
                    # Attnetion Heads
                    elif isinstance(module, torch.nn.Softmax):
                        self.latent_relevances[name] = (
                            self.latent_relevances[name]
                            .sum(dim=-1)
                            .transpose(1, 2)
                            .detach()
                            .cpu()
                        )
                        self.latent_activations[name] = (
                            self.latent_activations[name]
                            .sum(dim=-1)
                            .transpose(1, 2)
                            .detach()
                            .cpu()
                        )

    def register_hooks(self, model):
        """
        Attach hooks to the modules

        Args:
            model (torch.nn.module): model

        Returns:
            list: handles of the forward hooks
        """
        hook_handles = []
        for layer_name in self.layers_list_to_track:
            for name, module in model.named_modules():
                if name == layer_name:
                    hook_handles.append(
                        module.register_forward_hook(
                            self.get_hook_function(name, self.latent_output)
                        )
                    )

        return hook_handles

    @staticmethod
    def get_hook_function(layer_name, layer_out):
        """
        Static method to get the hook function

        Args:
            layer_name (str): layer_name
            layer_out (dict): dictionary to store the output of the layers
        """

        def forward_hook_function(module, input, output):
            # input in first layer is the input data
            # output in first layer is the output of the first layer
            layer_out[layer_name] = output
            output.retain_grad()

        return forward_hook_function

    def clear_latent_info(self):
        self.latent_output = {}
        self.latent_relevances = {}
        self.latent_activations = {}


class LXTLatentRelevanceAttributor(LatentRelevanceAttributor):
    def __init__(self, layers_list_to_track) -> None:
        """
        Constructor

        Args:
            layers_list_to_track (list): list of types of layers to track
        """
        super().__init__(layers_list_to_track)
        self.get_latent_activations = {}
        self.set_latent_activations = {}
        self.get_latent_relevances = {}

    def parse_latent_relevances(self, model):
        """
        Extract the relevance values from the output tensors
        via .grad

        Args:
            model (torch.nn.module): model
        """
        self.latent_relevances = {}
        self.latent_activations = {}
        for layer_name in self.layers_list_to_track:
            for name, module in model.named_modules():
                if name == layer_name:
                    # Sum the relevance over the spatial dimensions
                    # to acquire the concept's relevance
                    if isinstance(module, torch.nn.Conv2d):
                        self.latent_relevances[name] = (
                            self.get_latent_relevances[name]
                            .sum(dim=(2, 3))
                            .detach()
                            .cpu()
                        )
                        self.latent_activations[name] = (
                            self.get_latent_activations[name]
                            .sum(dim=(2, 3))
                            .detach()
                            .cpu()
                        )
                    elif True in [
                        isinstance(module, softmax_rules_types)
                        for softmax_rules_types in [
                            IxGSoftmaxBriefModule,
                            InputRefXGSoftmaxZPlusModuleVMAP,
                            BlockModule,
                        ]
                    ]:
                        self.latent_relevances[name] = (
                            self.get_latent_relevances[name]
                            .sum(dim=-1)
                            .transpose(1, 2)
                            .detach()
                            .cpu()
                        )
                        self.latent_activations[name] = (
                            self.get_latent_activations[name]
                            .sum(dim=-1)
                            .transpose(1, 2)
                            .detach()
                            .cpu()
                        )
                    elif isinstance(module, torch.nn.Linear):
                        self.latent_relevances[name] = (
                            self.get_latent_relevances[name].detach().cpu()
                        )
                        self.latent_activations[name] = (
                            self.get_latent_activations[name].detach().cpu()
                        )

    def register_hooks(self, model):
        hook_handles = []
        for layer_name in self.layers_list_to_track:
            for name, module in model.named_modules():
                if name == layer_name:
                    hook_handles.append(
                        module.register_backward_hook(
                            self.backward_get_hook_wrapper(layer_name)
                        )
                    )
                    hook_handles.append(
                        module.register_forward_hook(
                            self.forward_get_hook_wrapper(layer_name)
                        )
                    )

        return hook_handles

    def backward_get_hook_wrapper(self, layer_name):
        def get_relevance(module, in_gradient, out_gradient):
            # print("Backward hook executed")
            # print(f"LXT Activated hook at {layer_name}")
            self.get_latent_relevances[layer_name] = out_gradient[0].cpu()

        return get_relevance

    def forward_get_hook_wrapper(self, layer_name):
        def get_out_activations(module, input, output):
            # print("Forward hook executed")
            self.get_latent_activations[layer_name] = output.cpu()

        return get_out_activations

    def remove_hooks(self):
        """
        Remove the hooks
        """
        for handle in self.hook_handles:
            handle.remove()

    def clear_latent_info(self):
        self.get_latent_activations = {}
        self.get_latent_relevances = {}


class IntegratedGradientLatentAttributorZennit(ZennitLatentRelevanceAttributor):
    def __init__(self, layers_list_to_track) -> None:
        """
        Constructor

        Args:
            layers_list_to_track (list): list of types of layers to track
        """
        super().__init__(layers_list_to_track)
        self.latent_relevances = {}
        self.latent_activations = {}

    def lrp_pass(self, model, inputs, targets, composite, init_relevance, device):
        """
        Compute the relevance using Integrated Gradient

        Args:
            model (torch.nn.module): the model to be explained
            inputs (torch.tensor): inputs or the given images
            targets (torch.tensor): targets of the given images
            composite (): lrp composite
            device (): device to be used

        Returns:
            (torch.tensor): the computed heatmap using LRP
        """
        self.num_iter = 20
        self.iter_counter = 0
        with torch.enable_grad():
            self.clear_latent_info()
            self.hook_handles = self.register_hooks(model)

            def init_relevance(output_logits):
                grad_mask = torch.zeros_like(output_logits)
                grad_mask[:, targets] = 1.0

                return grad_mask

            baseline_fn = torch.zeros_like
            baseline = baseline_fn(inputs)
            result = torch.zeros_like(inputs)
            for alpha in torch.linspace(1.0 / self.num_iter, 1.0, self.num_iter):
                path_step = baseline + alpha * (inputs - baseline)

                if not path_step.requires_grad:
                    path_step.requires_grad = True
                output = model(path_step)
                (gradient,) = torch.autograd.grad(
                    (output,),
                    (path_step,),
                    grad_outputs=(init_relevance(output),),
                    create_graph=False,
                    retain_graph=None,
                )
                self.parse_latent_relevances(model)

                result += gradient / self.num_iter

            result *= inputs - baseline

            self.remove_hooks()

    def parse_latent_relevances(self, model):
        """
        Extract the relevance values from the output tensors
        via .grad

        Args:
            model (torch.nn.module): model
        """
        self.iter_counter += 1
        for layer_name in self.layers_list_to_track:
            for name, module in model.named_modules():
                if name == layer_name:
                    current_relevance = self.latent_output[name].grad.detach().cpu()
                    self.latent_output[name].detach().cpu()
                    self.latent_activations[name] = (
                        self.latent_output[name].detach().cpu()
                    )
                    # Sum the relevance over the spatial dimensions
                    # to acquire the concept's relevance
                    if isinstance(module, torch.nn.Linear):
                        if len(current_relevance.shape) == 3:
                            if name not in self.latent_relevances:
                                self.latent_relevances[name] = (
                                    current_relevance.detach().cpu()
                                )
                            else:
                                self.latent_relevances[
                                    name
                                ] += current_relevance.detach().cpu()
                            self.latent_activations[name] = (
                                self.latent_activations[name].detach().cpu()
                            )
                    elif isinstance(module, torch.nn.Conv2d):
                        # initialize the relevance if name does not exist
                        if name not in self.latent_relevances:
                            self.latent_relevances[name] = (
                                current_relevance.sum(dim=(2, 3)).detach().cpu()
                            )
                        else:
                            self.latent_relevances[name] += (
                                current_relevance.sum(dim=(2, 3)).detach().cpu()
                            )
                        self.latent_activations[name] = (
                            self.latent_activations[name].sum(dim=(2, 3)).detach().cpu()
                        )
                    # Attnetion Heads
                    elif isinstance(module, torch.nn.Softmax):
                        # initialize the relevance if name does not exist
                        if name not in self.latent_relevances:
                            self.latent_relevances[name] = (
                                current_relevance.sum(dim=-1)
                                .transpose(1, 2)
                                .detach()
                                .cpu()
                            )
                        # aggregate the relevance if name exists
                        else:
                            self.latent_relevances[name] += (
                                current_relevance.sum(dim=-1)
                                .transpose(1, 2)
                                .detach()
                                .cpu()
                            )
                        self.latent_activations[name] = (
                            self.latent_activations[name]
                            .sum(dim=-1)
                            .transpose(1, 2)
                            .detach()
                            .cpu()
                        )
                    if self.iter_counter == self.num_iter:
                        self.latent_relevances[name] /= self.num_iter


class ComponentAttibution:
    def __init__(self, attribution_type, model_type, target_layer_type):
        self.model_type = model_type
        self.attribution_type = attribution_type
        self.target_layer_type = target_layer_type
        self.attributor = self.choose_attributor()

    def choose_attributor(self):
        if self.model_type == "CNN":
            if self.target_layer_type == torch.nn.Conv2d:
                attributor = ZennitLatentRelevanceAttributor
        elif self.model_type == "ViT":
            if self.target_layer_type == torch.nn.Softmax:
                attributor = LXTLatentRelevanceAttributor
                if self.attribution_type in ["Relevance", "Random"]:
                    attributor = LXTLatentRelevanceAttributor
                elif self.attribution_type == "IntGrad":
                    attributor = IntegratedGradientLatentAttributorZennit
            elif self.target_layer_type == torch.nn.Linear:
                attributor = ZennitLatentRelevanceAttributor
                if self.attribution_type in ["Relevance", "Random"]:
                    attributor = ZennitLatentRelevanceAttributor
                elif self.attribution_type == "IntGrad":
                    attributor = IntegratedGradientLatentAttributorZennit

        return attributor

    @staticmethod
    def get_layer_names(model, target_layer_type):
        layer_names = ModelLayerUtils.get_layer_names(model, [target_layer_type])
        if target_layer_type == torch.nn.Softmax:
            layer_names = [name.replace(".module", "") for name in layer_names]
        # don't prune the last layer if the model is a linear model,
        # therefore we ignore it
        if target_layer_type == torch.nn.Linear:
            layer_names = layer_names[:-1]

        return layer_names

    def attribute(
        self, model, dataloader, attribution_composite, abs_flag=True, device="cpu"
    ):
        model.eval()
        # registering the composites + canonizers
        if self.attribution_type == "IntGrad":
            canonizer = get_vit_canonizer(["ReplaceAttention"])
            canonizer.apply(model)
        else:
            attribution_composite.register(model)

        # Currect layer names are obtained
        # after the composites + canonizers
        # are registered
        self.layer_names = ComponentAttibution.get_layer_names(
            model, self.target_layer_type
        )

        attributor = self.attributor(self.layer_names)

        sum_latent_relevances = OrderedDict([])
        for images, labels in dataloader:
            # Use composite=None because the composite
            # has been already registered to the model
            attributor.lrp_pass(
                model,
                images.to(device),
                labels.to(device),
                composite=None,
                # attribution_composite,
                initial_relevance=1,
                device=device,
            )

            for layer_name in self.layer_names:
                # Get latent relevances for each layer
                latent_relevance = (
                    attributor.latent_relevances[layer_name].detach().cpu()
                )

                if abs_flag:
                    latent_relevance = torch.abs(latent_relevance)

                if self.model_type == "CNN":
                    latent_relevance = latent_relevance.sum(dim=0)
                elif self.model_type == "ViT":
                    # Summing over the extra
                    # dimension of tokens
                    latent_relevance = latent_relevance.sum(dim=(0, 1))

                # Add the local latent relevance to the
                # corresponding dictionary
                if layer_name not in sum_latent_relevances.keys():
                    sum_latent_relevances[layer_name] = latent_relevance
                else:
                    sum_latent_relevances[layer_name] += latent_relevance

                # Taking care of Random Pruning
                if self.attribution_type == "Random":
                    sum_latent_relevances[layer_name] = torch.rand_like(
                        latent_relevance
                    )
                    break

        return sum_latent_relevances
