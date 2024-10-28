import torch.nn as nn
from pxp.LiT.core import (
    GenericRule,
    PassRule,
    DeepTaylorRule,
    BlockRule,
    IRefXGEpsilonRule,
    IRefXGJacobianGenericRule,
    IRefXGSoftmaxJacobianGenericRule,
    IRefXGMultiplyGenericRule,
    IRefXGClipRule,
)
from pxp.LiT.core import (
    AbsRule,
    stabilize,
    IRefXGSoftmaxEpsilonRule,
    IRefXGSoftmaxGenericRule,
)
import torch
import pxp.LiT.core as core

from zennit.rules import ClampMod, zero_bias, NoMod


class EpsilonModule(nn.Module):

    def __init__(self, module, zero_params=None):
        super().__init__()

        self.module = module

        self.input_modifiers = [lambda input: input]
        self.param_modifiers = [NoMod(zero_params=zero_params)]
        self.output_modifiers = [lambda output: output]
        self.modifiers = (
            self.input_modifiers,
            self.param_modifiers,
            self.output_modifiers,
        )

    def forward(self, *args, **kwargs):

        return GenericRule.apply(self.module, self.modifiers, kwargs, *args)


class ZPlusModule(nn.Module):

    def __init__(self, module, zero_params=None):
        super().__init__()

        self.module = module

        self.input_modifiers = [
            lambda input: input.clamp(min=0),
            lambda input: input.clamp(max=0),
        ]
        self.param_modifiers = [
            ClampMod(min=0.0, zero_params=zero_params),
            ClampMod(max=0.0, zero_params=zero_bias(zero_params)),
        ]
        self.output_modifiers = [lambda output: output] * 2
        self.modifiers = (
            self.input_modifiers,
            self.param_modifiers,
            self.output_modifiers,
        )

    def forward(self, *args, **kwargs):

        return GenericRule.apply(self.module, self.modifiers, kwargs, *args)


class AbsLRPModule(nn.Module):

    def __init__(self, module, zero_params=None):
        super().__init__()

        self.module = module

        self.input_modifiers = [
            lambda input: input.clamp(min=0),
            lambda input: input.clamp(max=0),
        ]
        self.param_modifiers = [
            ClampMod(min=0.0, zero_params=zero_params),
            ClampMod(max=0.0, zero_params=zero_bias(zero_params)),
        ]
        self.output_modifiers = [lambda output: abs(output)] * 2
        self.modifiers = (
            self.input_modifiers,
            self.param_modifiers,
            self.output_modifiers,
        )

    def forward(self, *args, **kwargs):

        return GenericRule.apply(self.module, self.modifiers, kwargs, *args)


class PassModule(nn.Module):

    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, *args, **kwargs):

        return PassRule.apply(self.module, kwargs, *args)


class BlockModule(nn.Module):

    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, *args, **kwargs):

        return BlockRule.apply(self.module, kwargs, *args)


class DeepTaylorModule(nn.Module):

    def __init__(self, module, root_fn, virtual_bias=False) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn
        self.virtual_bias = virtual_bias

    def forward(self, *args, **kwargs):

        return DeepTaylorRule.apply(
            self.module, self.root_fn, self.virtual_bias, kwargs, *args
        )


class LinearAlphaBeta(nn.Module):

    def __init__(self, module, alpha=2, beta=1) -> None:
        super().__init__()
        self.module = module
        self.options = {
            "alpha": alpha,
            "beta": beta,
        }

    def forward(self, *args, **kwargs):

        return core.LinearAlphaBetaRule.apply(self.module, self.options, kwargs, *args)


class LinearEpsilonStdMean(nn.Module):

    def __init__(self, module) -> None:
        super().__init__()
        self.module = module
        self.options = {}

    def forward(self, *args, **kwargs):

        return core.LinearEpsilonStdMeanRule.apply(
            self.module, self.options, kwargs, *args
        )


class InputRefXGradientEpsilonModule(nn.Module):

    def __init__(self, module, root_fn, virtual_bias=False) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn
        self.virtual_bias = virtual_bias

    def forward(self, *args, **kwargs):

        return IRefXGEpsilonRule.apply(
            self.module, self.root_fn, self.virtual_bias, kwargs, *args
        )


class IxGSoftmaxBriefModule(nn.Module):

    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):

        return core.IxGSoftmaxBriefRule.apply(self.module, kwargs, *args)


class IxGSoftmaxBriefMeanModule(nn.Module):

    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):

        return core.IxGSoftmaxBriefMeanRule.apply(self.module, kwargs, *args)


class InputRefXGradientGenericModule(nn.Module):

    def __init__(self, module, root_fn, modifier) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn
        self.modifier = modifier

    def forward(self, *args, **kwargs):

        return IRefXGJacobianGenericRule.apply(
            self.module, self.modifier, self.root_fn, kwargs, *args
        )


class InputRefXGradientZPlusModule(nn.Module):

    def __init__(self, module, root_fn) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn
        self.modifier = lambda input: input.clamp(min=0)

    def forward(self, *args, **kwargs):

        return IRefXGJacobianGenericRule.apply(
            self.module, self.modifier, self.root_fn, kwargs, *args
        )


class EfficientInputRefXGradientZPlusModule(nn.Module):

    def __init__(self, module, root_fn) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn
        self.modifier = lambda input: input.clamp(min=0)

    def forward(self, *args, **kwargs):

        return core.EfficientIRefXGJacobianGenericRule.apply(
            self.module, self.modifier, self.root_fn, kwargs, *args
        )


class InputRefXJacobianSoftmaxModule(nn.Module):

    def __init__(self, module, root_fn, modifier) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn
        self.modifier = modifier

    def forward(self, *args, **kwargs):

        return IRefXGSoftmaxJacobianGenericRule.apply(
            self.module, self.modifier, self.root_fn, kwargs, *args
        )


class InputRefXSoftmaxEpsilonModule(nn.Module):

    def __init__(self, module, root_fn) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn

    def forward(self, *args, **kwargs):

        return IRefXGSoftmaxEpsilonRule.apply(self.module, self.root_fn, kwargs, *args)


class InputRefXSoftmaxEpsilonAbsOuterOutModule(nn.Module):

    def __init__(self, module, root_fn) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn

    def forward(self, *args, **kwargs):

        return core.IRefXGSoftmaxEpsilonAbsOutRule.apply(
            self.module, self.root_fn, kwargs, *args
        )


class InputRefXGSoftmaxEpsilonAbsInnerOutModuleVMAP(nn.Module):

    def __init__(self, module, ref_fn, virtual_bias=True, chunk_size=32) -> None:
        super().__init__()
        self.module = module
        self.options = {
            "ref_fn": ref_fn,
            "jac_modifier": lambda x: x,
            "out_modifier": lambda x: x.abs_(),
            "virtual_bias": virtual_bias,
            "chunk_size": chunk_size,
        }

    def forward(self, *args, **kwargs):

        return core.IRefXGSoftmaxInnerRuleVMAP.apply(
            self.module, self.options, kwargs, *args
        )


class InputRefXSoftmaxZPlusModule(nn.Module):

    def __init__(self, module, root_fn) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn

        input_mod = [lambda input: input.clamp(min=0), lambda input: input.clamp(max=0)]
        jac_mod = input_mod
        output_mod = lambda output: sum(output)
        gradient_mod = (
            lambda grad_outputs, outputs: [grad_outputs / stabilize(outputs)] * 2
        )
        relevance_mod = lambda relevance: sum(relevance)

        self.modifiers = (input_mod, jac_mod, output_mod, gradient_mod, relevance_mod)

    def forward(self, *args, **kwargs):

        return IRefXGSoftmaxGenericRule.apply(
            self.module, self.modifiers, self.root_fn, kwargs, *args
        )


class InputRefXGSoftmaxEpsilonModuleVMAP(nn.Module):

    def __init__(self, module, ref_fn, virtual_bias=True, chunk_size=32) -> None:
        super().__init__()
        self.module = module
        self.options = {
            "ref_fn": ref_fn,
            "modifier": lambda x: x,
            "virtual_bias": virtual_bias,
            "chunk_size": chunk_size,
        }

    def forward(self, *args, **kwargs):

        return core.IRefXGSoftmaxRuleVMAP.apply(
            self.module, self.options, kwargs, *args
        )


class InputRefXGSoftmaxZPlusModuleVMAP(nn.Module):

    def __init__(self, module, ref_fn, virtual_bias=True, chunk_size=32) -> None:
        super().__init__()
        self.module = module
        self.options = {
            "ref_fn": ref_fn,
            "modifier": lambda x: torch.nn.functional.relu(x, inplace=True),
            "virtual_bias": virtual_bias,
            "chunk_size": chunk_size,
        }

    def forward(self, *args, **kwargs):

        return core.IRefXGSoftmaxRuleVMAP.apply(
            self.module, self.options, kwargs, *args
        )


class SoftmaxValueIxGVMAPModule(nn.Module):

    def __init__(self, module, virtual_bias=True, chunk_size=1) -> None:
        super().__init__()
        self.module = module
        self.options = {
            "modifier": lambda x: x,
            "virtual_bias": virtual_bias,
            "chunk_size": chunk_size,
        }

    def forward(self, *args, **kwargs):

        return core.SoftmaxValueIxGVMAPRule.apply(
            self.module, self.options, kwargs, *args
        )


class SoftmaxValueIxGZPlusVMAPModule(nn.Module):

    def __init__(self, module, virtual_bias=False, chunk_size=1) -> None:
        super().__init__()
        self.module = module
        self.options = {
            "modifier": lambda x: torch.nn.functional.relu(x, inplace=True),
            "virtual_bias": virtual_bias,
            "chunk_size": chunk_size,
        }

    def forward(self, *args, **kwargs):

        return core.SoftmaxValueIxGVMAPRule.apply(
            self.module, self.options, kwargs, *args
        )


class SoftmaxValueDTPassZPlusModule(nn.Module):

    def __init__(self, module) -> None:
        super().__init__()
        self.module = module
        self.options = {
            "modifier": lambda x: torch.nn.functional.relu(x, inplace=False),
        }

    def forward(self, *args, **kwargs):

        return core.SoftmaxValueDTApproxRule.apply(
            self.module, self.options, kwargs, *args
        )


class SoftmaxDTPassEpsilonModule(nn.Module):

    def __init__(self, module, virtual_bias=False, abs_out=False) -> None:
        super().__init__()
        self.module = module
        self.options = {
            "virtual_bias": virtual_bias,
            "modifier": lambda x: x,
            "abs": abs_out,
        }

    def forward(self, *args, **kwargs):

        return core.SoftmaxDTPassRule.apply(self.module, self.options, kwargs, *args)


class SoftmaxDTPassZPlusModule(nn.Module):

    def __init__(self, module, virtual_bias=False) -> None:
        super().__init__()
        self.module = module
        self.options = {
            "virtual_bias": virtual_bias,
            "modifier": lambda x: torch.nn.functional.relu(x, inplace=False),
            "abs": False,
        }

    def forward(self, *args, **kwargs):

        return core.SoftmaxDTPassRule.apply(self.module, self.options, kwargs, *args)


class SoftmaxIxGPassZPlusModule(nn.Module):

    def __init__(self, module, virtual_bias=False) -> None:
        super().__init__()
        self.module = module
        self.options = {
            "virtual_bias": virtual_bias,
            "modifier": lambda x: torch.nn.functional.relu(x, inplace=False),
        }

    def forward(self, *args, **kwargs):

        return core.SoftmaxIxGPassRule.apply(self.module, self.options, kwargs, *args)


class SoftmaxIxGPassModule(nn.Module):

    def __init__(self, module, virtual_bias=False) -> None:
        super().__init__()
        self.module = module
        self.options = {
            "virtual_bias": virtual_bias,
            "modifier": lambda x: x,
        }

    def forward(self, *args, **kwargs):

        return core.SoftmaxIxGPassRule.apply(self.module, self.options, kwargs, *args)


class SoftmaxDTPassConditionalModule(nn.Module):

    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):

        return core.SoftmaxDTConditionalPassRule.apply(self.module, {}, kwargs, *args)


class SoftmaxStablePassModule(nn.Module):

    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):

        return core.SoftmaxStablePassRule.apply(self.module, {}, kwargs, *args)


class InputRefXGSoftmaxAlphaBetaModule(nn.Module):

    def __init__(self, module, root_fn, alpha=2, beta=1) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn

        input_mod = [
            lambda input: input.clamp(min=0),
            lambda input: input.clamp(max=0),
            lambda input: input.clamp(min=0),
            lambda input: input.clamp(max=0),
        ]
        jac_mod = [
            lambda input: input.clamp(min=0),
            lambda input: input.clamp(max=0),
            lambda input: input.clamp(max=0),
            lambda input: input.clamp(min=0),
        ]
        output_mod = lambda output: [sum(output[:2])] * 2 + [sum(output[2:])] * 2

        gradient_mod = lambda grad_outputs, outputs: [
            grad_outputs / stabilize(out) for out in outputs
        ]
        relevance_mod = lambda relevance: alpha * sum(relevance[:2]) - beta * sum(
            relevance[2:]
        )

        self.modifiers = (input_mod, jac_mod, output_mod, gradient_mod, relevance_mod)

    def forward(self, *args, **kwargs):

        return IRefXGSoftmaxGenericRule.apply(
            self.module, self.modifiers, self.root_fn, kwargs, *args
        )


class GradientxInputReferenceModule(nn.Module):

    def __init__(self, module, ref_fn, bias, distribute_bias) -> None:
        super().__init__()
        self.module = module
        self.ref_fn = ref_fn
        self.bias = bias
        self.distribute_bias = distribute_bias

    def forward(self, *args, **kwargs):
        return core.GradientxInputReference.apply(
            self.module, self.ref_fn, self.bias, self.distribute_bias, kwargs, *args
        )


class DeepTaylorReferenceModule(nn.Module):

    def __init__(self, module, ref_fn, bias, distribute_bias, clip) -> None:
        super().__init__()
        self.module = module
        self.ref_fn = ref_fn
        self.bias = bias
        self.distribute_bias = distribute_bias
        self.clip = clip

    def forward(self, *args, **kwargs):
        return core.DeepTaylorReference.apply(
            self.module,
            self.ref_fn,
            self.bias,
            self.distribute_bias,
            self.clip,
            kwargs,
            *args
        )


class InputRefXGMultiplyEpsilonModule(nn.Module):

    def __init__(self, module, root_fn, transpose) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn
        self.transpose = transpose

        self.mod_input = [lambda input: input]
        self.mod_jac = [lambda input: input]
        self.mod_output = lambda output: sum(output)
        self.mod_mask = lambda grad_outputs, outputs: [
            grad_outputs / stabilize(outputs)
        ]
        self.mod_relevance = lambda relevance: sum(relevance)

        self.modifiers = (
            self.mod_input,
            self.mod_jac,
            self.mod_output,
            self.mod_mask,
            self.mod_relevance,
        )

    def forward(self, *args, **kwargs):

        return IRefXGMultiplyGenericRule.apply(
            self.module, self.modifiers, self.root_fn, self.transpose, kwargs, *args
        )


class InputRefXGMultiplyEpsilonModuleVMAP(nn.Module):

    def __init__(
        self, module, ref_fn, transpose, virtual_bias=True, chunk_size=128
    ) -> None:
        super().__init__()
        self.module = module
        self.options = {
            "ref_fn": ref_fn,
            "transpose": transpose,
            "modifier": lambda x: x,
            "virtual_bias": virtual_bias,
            "chunk_size": chunk_size,
        }

    def forward(self, *args, **kwargs):

        return core.IRefXGMultiplyVMAPRule.apply(
            self.module, self.options, kwargs, *args
        )


class InputRefXGMultiplyZPlusModuleVMAP(nn.Module):

    def __init__(
        self, module, ref_fn, transpose, virtual_bias=True, chunk_size=128
    ) -> None:
        super().__init__()
        self.module = module
        self.options = {
            "ref_fn": ref_fn,
            "transpose": transpose,
            "modifier": lambda x: torch.nn.functional.relu(x, inplace=True),
            "virtual_bias": virtual_bias,
            "chunk_size": chunk_size,
        }

    def forward(self, *args, **kwargs):

        return core.IRefXGMultiplyVMAPRule.apply(
            self.module, self.options, kwargs, *args
        )


class InputRefXGMSoftmaxMulEpsilonModuleVMAP(nn.Module):

    def __init__(self, module, ref_fn, virtual_bias=True, chunk_size=32) -> None:
        super().__init__()
        self.module = module
        self.options = {
            "ref_fn": ref_fn,
            "modifier": lambda x: x,
            "virtual_bias": virtual_bias,
            "chunk_size": chunk_size,
        }

    def forward(self, *args, **kwargs):

        return core.IRefXGSoftmaxValueMulRuleVMAP.apply(
            self.module, self.options, kwargs, *args
        )


class InputRefXGMSoftmaxMulZPlusModuleVMAP(nn.Module):

    def __init__(self, module, ref_fn, virtual_bias=True, chunk_size=32) -> None:
        super().__init__()
        self.module = module
        self.options = {
            "ref_fn": ref_fn,
            "modifier": lambda x: torch.nn.functional.relu(x, inplace=True),
            "virtual_bias": virtual_bias,
            "chunk_size": chunk_size,
        }

    def forward(self, *args, **kwargs):

        return core.IRefXGSoftmaxValueMulRuleVMAP.apply(
            self.module, self.options, kwargs, *args
        )


class InputRefXGMultiplyZPlusModule(nn.Module):

    def __init__(self, module, root_fn, transpose) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn
        self.transpose = transpose

        input_mod = [lambda input: input.clamp(min=0), lambda input: input.clamp(max=0)]
        jac_mod = input_mod
        output_mod = lambda output: sum(output)
        gradient_mod = (
            lambda grad_outputs, outputs: [grad_outputs / stabilize(outputs)] * 2
        )
        relevance_mod = lambda relevance: sum(relevance)

        self.modifiers = (input_mod, jac_mod, output_mod, gradient_mod, relevance_mod)

    def forward(self, *args, **kwargs):

        return IRefXGMultiplyGenericRule.apply(
            self.module, self.modifiers, self.root_fn, self.transpose, kwargs, *args
        )


class InputRefXGMultiplyAlphaBetaModule(nn.Module):

    def __init__(self, module, root_fn, transpose, alpha=2, beta=1) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn
        self.transpose = transpose

        input_mod = [
            lambda input: input.clamp(min=0),
            lambda input: input.clamp(max=0),
            lambda input: input.clamp(min=0),
            lambda input: input.clamp(max=0),
        ]
        jac_mod = [
            lambda input: input.clamp(min=0),
            lambda input: input.clamp(max=0),
            lambda input: input.clamp(max=0),
            lambda input: input.clamp(min=0),
        ]
        output_mod = lambda output: [sum(output[:2])] * 2 + [sum(output[2:])] * 2

        gradient_mod = lambda grad_outputs, outputs: [
            grad_outputs / stabilize(out) for out in outputs
        ]
        relevance_mod = lambda relevance: alpha * sum(relevance[:2]) - beta * sum(
            relevance[2:]
        )

        self.modifiers = (input_mod, jac_mod, output_mod, gradient_mod, relevance_mod)

    def forward(self, *args, **kwargs):

        return IRefXGMultiplyGenericRule.apply(
            self.module, self.modifiers, self.root_fn, self.transpose, kwargs, *args
        )


class InputRefXGradientMultiplyGammaModuleOLD(nn.Module):

    def __init__(self, module, root_fn, transpose, gamma=1) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn
        self.transpose = transpose
        self.gamma = gamma

        self.mod_input = [
            lambda input: input,
            lambda input: input.clamp(min=0),
            lambda input: input.clamp(max=0),
        ]
        self.mod_jac = [
            lambda input: input,
            lambda input: input.clamp(min=0),
            lambda input: input.clamp(max=0),
        ]

        def mod_output(outputs: list):

            return outputs[0] + self.gamma * (outputs[1] + outputs[2])

        self.mod_output = mod_output
        self.mod_relevance = mod_output

        self.modifiers = (
            self.mod_input,
            self.mod_jac,
            self.mod_output,
            self.mod_relevance,
        )

    def forward(self, *args, **kwargs):

        return IRefXGMultiplyGenericRuleOLD.apply(
            self.module, self.modifiers, self.root_fn, self.transpose, kwargs, *args
        )


class InputRefXGradientMultiplyGammaModuleProOLD(nn.Module):

    def __init__(self, module, root_fn, transpose, gamma=1) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn
        self.transpose = transpose
        self.gamma = gamma

        self.mod_input = [
            lambda input: input,
            lambda input: input.clamp(min=0),
            lambda input: input.clamp(max=0),
            lambda input: input.clamp(min=0),
            lambda input: input.clamp(max=0),
        ]
        self.mod_jac = [
            lambda input: input,
            lambda input: input.clamp(min=0),
            lambda input: input.clamp(max=0),
            lambda input: input.clamp(max=0),
            lambda input: input.clamp(min=0),
        ]

        def mod_output(outputs: list):

            gamma_term = torch.where(
                outputs[0] >= 0, outputs[1] + outputs[2], outputs[3] + outputs[4]
            )
            return outputs[0] + self.gamma * gamma_term

        self.mod_output = mod_output

        def mod_mask(grad_output, output):

            return [
                grad_output,
                grad_output / stabilize(output.clamp(min=0)),
                grad_output / stabilize(output.clamp(min=0)),
                grad_output / stabilize(output.clamp(max=0)),
                grad_output / stabilize(output.clamp(max=0)),
            ]

        self.mod_relevance = mod_output

        self.modifiers = (
            self.mod_input,
            self.mod_jac,
            self.mod_output,
            self.mod_relevance,
        )

    def forward(self, *args, **kwargs):

        return IRefXGMultiplyGenericRuleOLD.apply(
            self.module, self.modifiers, self.root_fn, self.transpose, kwargs, *args
        )


class InputRefXGradientClipModule(nn.Module):

    def __init__(self, module, root_fn, factor=2) -> None:
        super().__init__()
        self.module = module
        self.root_fn = root_fn
        self.factor = factor

    def forward(self, *args, **kwargs):

        return IRefXGClipRule.apply(
            self.module, self.root_fn, self.factor, kwargs, *args
        )


class AbsModule(nn.Module):

    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, *args, **kwargs):

        return AbsRule.apply(self.module, kwargs, *args)


class GradientxInputReferenceModule(nn.Module):

    def __init__(self, module, ref_fn, bias, distribute_bias) -> None:
        super().__init__()
        self.module = module
        self.ref_fn = ref_fn
        self.bias = bias
        self.distribute_bias = distribute_bias

    def forward(self, *args, **kwargs):
        return core.GradientxInputReference.apply(
            self.module, self.ref_fn, self.bias, self.distribute_bias, kwargs, *args
        )


class DeepTaylorReferenceModule(nn.Module):

    def __init__(self, module, ref_fn, bias, distribute_bias, clip) -> None:
        super().__init__()
        self.module = module
        self.ref_fn = ref_fn
        self.bias = bias
        self.distribute_bias = distribute_bias
        self.clip = clip

    def forward(self, *args, **kwargs):
        return core.DeepTaylorReference.apply(
            self.module,
            self.ref_fn,
            self.bias,
            self.distribute_bias,
            self.clip,
            kwargs,
            *args
        )


class ElementwiseMultiplyCPModule(nn.Module):

    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, *args, **kwargs):

        return core.ElementwiseMultiplyCPRule.apply(self.module, kwargs, *args)


class LinearEpsilonQuantizeModule(nn.Module):

    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, *args, **kwargs):

        return core.LinearEpsilonQuantizeRule.apply(self.module, kwargs, *args)


class SumZPlusRule(nn.Module):

    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, *args, **kwargs):

        return core.SumZPlus.apply(self.module, kwargs, *args)
