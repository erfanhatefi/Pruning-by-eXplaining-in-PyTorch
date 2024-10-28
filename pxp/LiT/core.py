import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.func import jvp, vjp, jacrev, vmap
import math
from zennit.core import ParamMod
import torch.nn.functional as F

####
import pydevd

# pydevd.settrace(suspend=False, trace_only_current_thread=True)
###


def stabilize(input, epsilon=1e-6, clip=False, norm_scale=False, dim=None):

    sign = (input == 0.0).to(input) + input.sign()
    if norm_scale:
        if dim is None:
            dim = tuple(range(1, input.ndim))
        epsilon = epsilon * ((input**2).mean(dim=dim, keepdim=True) ** 0.5)
    if clip:
        return sign * input.abs().clip(min=epsilon)
    return input + sign * epsilon


class GenericRule(Function):

    @staticmethod
    def forward(ctx, module, modifiers, kwargs, *args):
        ctx.save_for_backward(*args)
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.modifiers = modifiers

        with torch.no_grad():  # TODO: seems not to be necessary
            outputs = module(*args, **kwargs)

        return outputs

    @staticmethod
    @once_differentiable  # TODO: not necessary?
    def backward(ctx, *grad_outputs):

        modifiers_input, modifiers_param, modifiers_output = ctx.modifiers

        n_outputs = len(grad_outputs)
        n_inputs = len(ctx.saved_tensors)
        n_backward = len(modifiers_input)

        relevances = []
        inputs, outputs = [], []

        for mod_in, mod_param, mod_out in zip(
            modifiers_input, modifiers_param, modifiers_output
        ):

            # detach is necessary to avoid superimposed gradients if the same input tensor is used multiple times
            inputs_t = tuple(
                mod_in(input).detach().requires_grad_() for input in ctx.saved_tensors
            )

            with ParamMod.ensure(mod_param)(
                ctx.module
            ) as modified, torch.enable_grad():
                outputs_t = modified(*inputs_t, **ctx.saved_kwargs)
                if not isinstance(outputs_t, tuple):
                    outputs_t = (outputs_t,)
                outputs_t = [mod_out(output) for output in outputs_t]

            inputs.append(inputs_t)
            outputs.append(outputs_t)

        output_sum = [
            sum(x) for x in zip(*outputs)
        ]  # sum over tuples i.e. outputs[0][0] + outputs[1][0]
        grad_masks = tuple(
            grad_outputs[i] / stabilize(output_sum[i]) for i in range(n_outputs)
        )

        for i in range(n_backward):
            grads = torch.autograd.grad(
                outputs[i], inputs[i], grad_masks
            )  # TODO: allow_unused=True
            relevance_t = tuple(
                grads[k] * inputs[i][k] if grads[k] != None else None
                for k in range(n_inputs)
            )
            relevances.append(relevance_t)

            if any(torch.isnan(tensor).any() for tensor in relevance_t):
                raise ValueError("NaN encountered")

        return (None, None, None) + tuple(sum(r) for r in zip(*relevances))


class SingleGenericRule(Function):

    @staticmethod
    def forward(ctx, module, modifiers, kwargs, *args):
        ctx.save_for_backward(*args)
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.modifiers = modifiers

        with torch.no_grad():  # TODO: seems not to be necessary
            outputs = module(*args, **kwargs)

        return outputs

    @staticmethod
    @once_differentiable  # TODO: not necessary?
    def backward(ctx, *grad_outputs):

        modifiers_input, modifiers_param, modifiers_output = ctx.modifiers
        modifier_gradient, modifier_reducer = ctx.modifiers[3:]  # TODO

        inputs, outputs = [], []

        for mod_in, mod_param, mod_out in zip(
            modifiers_input, modifiers_param, modifiers_output
        ):

            # detach is necessary to avoid superimposed gradients if the same input tensor is used multiple times
            inputs_t = tuple(
                mod_in(input).detach().requires_grad_() for input in ctx.saved_tensors
            )

            with ParamMod.ensure(mod_param)(
                ctx.module
            ) as modified, torch.enable_grad():
                outputs_t = modified(*inputs_t, **ctx.saved_kwargs)
                if not isinstance(outputs_t, tuple):
                    outputs_t = (outputs_t,)
                outputs_t = [mod_out(output) for output in outputs_t]

            inputs.append(inputs_t)
            outputs.append(outputs_t)

        grad_masks = modifier_gradient(
            grad_outputs[0], outputs
        )  # TODO: tuple(grad_outputs[i] / stabilize(output_sum[i]) for i in range(n_outputs))

        gradients = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=grad_masks,
        )
        relevance = modifier_reducer(inputs, gradients)

        return (None, None, None) + tuple(relevance)


class PassRule(Function):

    @staticmethod
    def forward(ctx, module, kwargs, *args):

        with torch.no_grad():
            outputs = module(*args, **kwargs)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        return (
            None,
            None,
        ) + grad_outputs


class BlockRule(Function):

    @staticmethod
    def forward(ctx, module, kwargs, *args):

        ctx.save_for_backward(*args)

        with torch.no_grad():
            outputs = module(*args, **kwargs)

        return outputs

    @staticmethod
    # @once_differentiable
    def backward(ctx, *grad_outputs):

        return (
            None,
            None,
        ) + tuple(
            torch.zeros_like(ctx.saved_tensors[i])
            for i in range(len(ctx.saved_tensors))
        )


class AbsMod(ParamMod):

    def __init__(self, **kwargs):
        def modifier(param, name):
            return abs(param)

        super().__init__(modifier, **kwargs)


class AbsRule(Function):

    @staticmethod
    def forward(ctx, module, kwargs, *args):
        ctx.save_for_backward(*args)
        ctx.saved_kwargs = kwargs
        ctx.module = module

        with torch.no_grad():  # TODO: seems not to be necessary
            outputs = module(*args, **kwargs)

        if isinstance(outputs, tuple):
            ctx.sign = tuple((out == 0.0).to(out) + out.sign() for out in outputs)
        else:
            ctx.sign = ((outputs == 0.0).to(outputs) + outputs.sign(),)

        return outputs

    @staticmethod
    @once_differentiable  # TODO: not necessary?
    def backward(ctx, *grad_outputs):

        mod_in = lambda x: x.abs()
        mod_param = AbsMod()

        n_outputs = len(grad_outputs)
        n_inputs = len(ctx.saved_tensors)

        # compute output
        inputs_t = tuple(
            mod_in(input).detach().requires_grad_() for input in ctx.saved_tensors
        )

        with ParamMod.ensure(mod_param)(ctx.module) as modified:
            outputs_t = modified(*inputs_t, **ctx.saved_kwargs)
            if not isinstance(outputs_t, tuple):
                outputs_t = (outputs_t,)

        grad_masks = tuple(
            grad_outputs[i] / stabilize(outputs_t[i] * ctx.sign[i], 1e-10)
            for i in range(n_outputs)
        )

        # compute vJ product
        inputs_t = tuple(input.detach().requires_grad_() for input in ctx.saved_tensors)
        with torch.enable_grad():
            outputs_t = ctx.module(*inputs_t, **ctx.saved_kwargs)
        if not isinstance(outputs_t, tuple):
            outputs_t = (outputs_t,)

        grads = torch.autograd.grad(outputs_t, inputs_t, grad_masks)
        relevances = tuple(
            grads[k] * inputs_t[k] if grads[k] != None else None
            for k in range(n_inputs)
        )

        return (None, None) + relevances


class DeepTaylorRule(Function):

    @staticmethod
    def forward(ctx, module, root_fn, virtual_bias, kwargs, *args):
        ctx.save_for_backward(*args)
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.virtual_bias = virtual_bias

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)
            ctx.root = root_fn(*args)

            assert isinstance(ctx.root, tuple)
            ctx.outputs = outputs.detach()

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        inputs_r = tuple(
            ctx.saved_tensors[i] - ctx.root[i] for i in range(len(ctx.saved_tensors))
        )

        def myfunc(*args):
            return ctx.module(*args, **ctx.saved_kwargs)

        # ctx.root and inputs_r must have same shapes
        values, Jvs = jvp(myfunc, ctx.root, inputs_r)

        if isinstance(values, tuple):
            if ctx.virtual_bias:
                outputs = ctx.outputs
            else:
                outputs = tuple(values[i] + Jvs[i] for i in range(len(values)))
            grad_masks = tuple(
                grad_outputs[i] / stabilize(outputs[i]) for i in range(len(outputs))
            )
        else:
            if ctx.virtual_bias:
                outputs = ctx.outputs
            else:
                outputs = values + Jvs
            grad_masks = grad_outputs[0] / stabilize(outputs)

        _, vjpfunc = vjp(myfunc, *ctx.root)
        grads = vjpfunc(grad_masks)

        return (None, None, None, None) + tuple(
            grads[i] * inputs_r[i] if grads[i] != None else None
            for i in range(len(inputs_r))
        )


class IRefXGEpsilonRule(Function):

    @staticmethod
    def forward(ctx, module, root_fn, virtual_bias, kwargs, *args):
        ctx.save_for_backward(*args)
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.virtual_bias = virtual_bias

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)
            ctx.root = root_fn(*args)

            assert isinstance(ctx.root, tuple)
            ctx.outputs = outputs.detach().clone()

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        inputs_r = tuple(
            ctx.saved_tensors[i] - ctx.root[i] for i in range(len(ctx.saved_tensors))
        )

        def myfunc(*args):
            return ctx.module(*args, **ctx.saved_kwargs)

        # ctx.saved_tensors and inputs_r must have same shapes
        _, Jvs = jvp(myfunc, ctx.saved_tensors, inputs_r)

        if isinstance(Jvs, tuple):

            if ctx.virtual_bias:
                raise ValueError("virtual bias not implemented")
            else:
                outputs = tuple(Jvs[i] for i in range(len(Jvs)))

            grad_masks = tuple(
                grad_outputs[i] / stabilize(outputs[i]) for i in range(len(outputs))
            )
        else:
            if ctx.virtual_bias:
                outputs = ctx.outputs
            else:
                outputs = Jvs

            grad_masks = grad_outputs[0] / stabilize(outputs)

        _, vjpfunc = vjp(myfunc, *ctx.saved_tensors)
        grads = vjpfunc(grad_masks)

        ##
        # if isinstance(ctx.module, torch.nn.Softmax):
        #     relevance = tuple(grads[i]*inputs_r[i] if grads[i] != None else None for i in range(len(inputs_r)))
        #     print("rule ratio in/out", relevance[0].sum() / grad_outputs[0].sum())
        ##

        if torch.isnan(grads[0]).any():
            raise ValueError("NaN here")

        return (None, None, None, None) + tuple(
            grads[i] * inputs_r[i] if grads[i] != None else None
            for i in range(len(inputs_r))
        )


class IRefXGJacobianGenericRule(Function):

    @staticmethod
    def forward(ctx, module, modifier, root_fn, kwargs, *args):
        ctx.save_for_backward(*args)
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.modifier = modifier

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)
            ctx.root = root_fn(*args)

            assert isinstance(ctx.root, tuple)

        return outputs

    @staticmethod
    def modified_output(J, inputs, grad_output, modifier):
        """
        outputs = sum_i mod(J_ji * inputs_i) = mod(J * inputs) @ ones
        """

        batch_size = len(inputs)
        expanded_dim = [1] * len(grad_output.shape[1:])

        mod_hadamard = modifier(
            J * inputs.view(batch_size, *expanded_dim, *inputs.shape[1:])
        )

        ones = torch.ones_like(inputs).view(batch_size, -1)
        mod_hadamard = mod_hadamard.view(batch_size, *grad_output.shape[1:], -1)

        return torch.einsum("b...i,bi->b...", mod_hadamard, ones)

    @staticmethod
    def modified_relevance(J, inputs, grad_mask, modifier):
        """
        relevance = sum_j mod(J_ji * inputs_i) * R_j/outputs = mod(J * inputs).T @ R/outputs
        with grad_mask = R/outputs
        """

        batch_size = len(inputs)
        expanded_dim = [1] * len(grad_mask.shape[1:])

        mod_hadamard = modifier(
            J * inputs.view(batch_size, *expanded_dim, *inputs.shape[1:])
        )

        grad_mask = grad_mask.view(batch_size, -1)
        mod_hadamard = mod_hadamard.view(batch_size, -1, *inputs.shape[1:])

        return torch.einsum("bk,bk...->b...", grad_mask, mod_hadamard)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        n_inputs = len(ctx.saved_tensors)
        n_outputs = len(grad_outputs)
        inputs_r = tuple(ctx.saved_tensors[i] - ctx.root[i] for i in range(n_inputs))

        def myfunc(*args):
            return ctx.module(*args, **ctx.saved_kwargs)

        # linearize the function at the actual input point
        # jacobian tensor structure: (*output_shape, *input_shape)
        jacobian = vmap(jacrev(myfunc, argnums=tuple(n for n in range(n_inputs))))(
            *ctx.saved_tensors
        )

        # convert jacobian to consistent data structure for easier handling
        # jacobian tuple structure: (n_outputs, n_inputs)
        if n_outputs == 1:
            jacobian = (jacobian,)

        # compute outputs as jacobian vector product: jacobian @ (inputs-root)
        outputs = [torch.zeros_like(grad_outputs[k]) for k in range(n_outputs)]
        for k in range(n_outputs):
            for i in range(n_inputs):

                output = IRefXGJacobianGenericRule.modified_output(
                    jacobian[k][i], inputs_r[i], grad_outputs[k], ctx.modifier
                )
                outputs[k] += output

        # compute relevance / outputs (Rout.T/out.T)
        grad_masks = tuple(
            grad_outputs[k] / stabilize(outputs[k]) for k in range(len(outputs))
        )

        # compute relevance as vector jacobian product: (input.T-root.T) * (Rout.T/out.T @ jacobian)
        relevances = [torch.zeros_like(ctx.saved_tensors[i]) for i in range(n_inputs)]
        for i in range(n_inputs):
            for k in range(n_outputs):

                relevance = IRefXGJacobianGenericRule.modified_relevance(
                    jacobian[k][i], inputs_r[i], grad_masks[k], ctx.modifier
                )
                relevances[i] += relevance

        return (None, None, None, None) + tuple(relevances)


class EfficientIRefXGJacobianGenericRule(Function):

    @staticmethod
    def forward(ctx, module, modifier, root_fn, kwargs, *args):
        ctx.save_for_backward(*args)
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.modifier = modifier

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)
            ctx.root = root_fn(*args)

            assert isinstance(ctx.root, tuple)

        return outputs

    @staticmethod
    def modified_output(J, inputs, grad_output, modifier):
        """
        outputs = sum_i mod(J_ji * inputs_i) = mod(J * inputs) @ ones
        """

        batch_size = len(inputs)
        expanded_dim = [1] * len(grad_output.shape[1:])

        mod_hadamard = modifier(
            J * inputs.view(batch_size, *expanded_dim, *inputs.shape[1:])
        )

        ones = torch.ones_like(inputs).view(batch_size, -1)
        mod_hadamard = mod_hadamard.view(batch_size, *grad_output.shape[1:], -1)

        return torch.einsum("b...i,bi->b...", mod_hadamard, ones)

    @staticmethod
    def modified_relevance(J, inputs, grad_mask, modifier):
        """
        relevance = sum_j mod(J_ji * inputs_i) * R_j/outputs = mod(J * inputs).T @ R/outputs
        with grad_mask = R/outputs
        """

        batch_size = len(inputs)
        expanded_dim = [1] * len(grad_mask.shape[1:])

        mod_hadamard = modifier(
            J * inputs.view(batch_size, *expanded_dim, *inputs.shape[1:])
        )

        grad_mask = grad_mask.view(batch_size, -1)
        mod_hadamard = mod_hadamard.view(batch_size, -1, *inputs.shape[1:])

        return torch.einsum("bk,bk...->b...", grad_mask, mod_hadamard)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        n_inputs = len(ctx.saved_tensors)
        inputs_r = tuple(ctx.saved_tensors[i] - ctx.root[i] for i in range(n_inputs))

        def myfunc(*args):
            return ctx.module(*args, **ctx.saved_kwargs)

        relevance = vmap(
            EfficientIRefXGJacobianGenericRule.compute_relevance,
            in_dims=(None, None, None, 0, None, 0),
            chunk_size=32,
        )(myfunc, n_inputs, inputs_r, ctx.saved_tensors, ctx.modifier, grad_outputs[0])

        return (None, None, None, None) + relevance

    @staticmethod
    @torch.no_grad()
    def compute_relevance(myfunc, n_inputs, inputs_r, inputs, modifier, grad_outputs):

        # linearize the function at the actual input point
        # jacobian tensor structure: (*input_shape)
        jacobian = jacrev(myfunc, argnums=tuple(n for n in range(n_inputs)))(*inputs)

        # compute outputs as jacobian vector product: jacobian @ (inputs-root)
        for i in range(n_inputs):

            output = EfficientIRefXGJacobianGenericRule.modified_output(
                jacobian[i], inputs_r[i], grad_outputs, modifier
            )

        # compute relevance / outputs (Rout.T/out.T)
        grad_masks = grad_outputs / stabilize(output)

        # compute relevance as vector jacobian product: (input.T-root.T) * (Rout.T/out.T @ jacobian)
        relevances = [torch.zeros_like(inputs[i]) for i in range(n_inputs)]
        for i in range(n_inputs):

            relevance = EfficientIRefXGJacobianGenericRule.modified_relevance(
                jacobian[i], inputs_r[i], grad_masks, modifier
            )
            relevances[i] += relevance

        return tuple(relevances)


class IxGLinearAdaptGammaRule(Function):

    @staticmethod
    def forward(ctx, module, factor, kwargs, *args):
        ctx.save_for_backward(*args)
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.factor = factor

        with torch.no_grad():  # TODO: seems not to be necessary
            outputs = module(*args, **kwargs)

        return outputs

    @staticmethod
    @once_differentiable  # TODO: not necessary?
    def backward(ctx, *grad_outputs):

        inputs = ctx.saved_tensors[0]

        input_shape = inputs.shape
        inputs = inputs.view(-1, inputs.shape[-1])
        grad_outputs = grad_outputs[0].view(-1, grad_outputs[0].shape[-1])

        W = ctx.module.weight
        b = ctx.module.bias
        if b is None:
            b = torch.zeros(W.shape[0])

        w_inp = W[None, :, :] * inputs[:, None, :]
        w_inp_pos = torch.clamp(w_inp, min=0.0)
        w_inp_neg = torch.clamp(w_inp, max=0.0)

        out = w_inp.sum(-1) + b[None, :]
        out_pos = w_inp_pos.sum(-1)
        out_neg = w_inp_neg.sum(-1)

        abs_max = w_inp.abs().max(dim=-1)[0]

        gamma = (ctx.factor * out) / stabilize(
            abs_max
            - (out > 0) * out_pos * ctx.factor
            - (out < 0) * out_neg * ctx.factor
        )
        # gamma = abs(gamma)

        out_final = out + (out > 0) * out_pos * gamma + (out < 0) * out_neg * gamma

        w_inp_final = (
            (out > 0)[:, :, None] * w_inp_pos * gamma[:, :, None]
            + (out < 0)[:, :, None] * w_inp_neg * gamma[:, :, None]
            + w_inp
        )

        ###
        # relevance_out = grad_outputs / stabilize(out)
        # relevance_inp = (w_inp * relevance_out[:, :, None]).sum(1)

        # return (None, None, None) + (relevance_inp.view(*input_shape),)
        ###

        relevance_out = grad_outputs / stabilize(out_final)

        relevance_inp = w_inp_final * relevance_out[:, :, None]
        relevance_inp = relevance_inp.sum(1)

        return (None, None, None) + (relevance_inp.view(*input_shape),)


class LinearAlphaBetaRule(Function):

    @staticmethod
    def forward(ctx, module, options, kwargs, *args):
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.options = options

        with torch.no_grad():  # TODO: seems not to be necessary
            outputs = module(*args, **kwargs)
            ctx.save_for_backward(*args, outputs.detach())

        return outputs

    @staticmethod
    @once_differentiable  # TODO: not necessary?
    def backward(ctx, *grad_outputs):

        inputs = ctx.saved_tensors[0]
        output = ctx.saved_tensors[1]

        input_shape = inputs.shape
        inputs = inputs.view(-1, inputs.shape[-1])
        output = output.view(-1, output.shape[-1])
        grad_outputs = grad_outputs[0].view(-1, grad_outputs[0].shape[-1])

        W = ctx.module.weight
        b = ctx.module.bias
        if b is None:
            b = torch.zeros(W.shape[0])

        w_inp = W[None, :, :] * inputs[:, None, :]
        w_inp_pos = torch.clamp(w_inp, min=0.0)
        w_inp_neg = torch.clamp(w_inp, max=0.0)

        out_pos = w_inp_pos.sum(-1) + torch.clamp(b[None, :], min=0.0)
        out_neg = w_inp_neg.sum(-1) + torch.clamp(b[None, :], max=0.0)

        relevance_out_pos = grad_outputs / stabilize(out_pos)
        relevance_out_neg = grad_outputs / stabilize(out_neg)

        relevance_pos = w_inp_pos * relevance_out_pos[:, :, None]
        relevance_neg = w_inp_neg * relevance_out_neg[:, :, None]
        relevance = (
            relevance_pos.sum(1) * ctx.options["alpha"]
            - relevance_neg.sum(1) * ctx.options["beta"]
        )

        assert ctx.options["alpha"] - ctx.options["beta"] == 1

        return (None, None, None) + (relevance.view(*input_shape),)


class LinearEpsilonStdMeanRule(Function):

    @staticmethod
    def forward(ctx, module, options, kwargs, *args):
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.options = options

        with torch.no_grad():  # TODO: seems not to be necessary
            outputs = module(*args, **kwargs)
            ctx.save_for_backward(*args, outputs.detach())

        return outputs

    @staticmethod
    @once_differentiable  # TODO: not necessary?
    def backward(ctx, *grad_outputs):

        inputs = ctx.saved_tensors[0]
        output = ctx.saved_tensors[1]

        input_shape = inputs.shape
        inputs = inputs.view(-1, inputs.shape[-1])
        output = output.view(-1, output.shape[-1])
        grad_outputs = grad_outputs[0].view(-1, grad_outputs[0].shape[-1])

        W = ctx.module.weight
        # W = (W - W.mean()) / W.std()
        W = W / W.std()
        b = ctx.module.bias
        if b is None:
            b = torch.zeros(W.shape[0])
        else:
            # b = (b - b.mean()) / b.std()
            b = b / b.std()

        w_inp = W[None, :, :] * inputs[:, None, :]
        out = w_inp.sum(-1) + b[None, :]

        relevance_out = grad_outputs / stabilize(out)

        relevance = (w_inp * relevance_out[:, :, None]).sum(1)

        return (None, None, None) + (relevance.view(*input_shape),)


class IRefXGMultiplyAdaptiveGammaVMAPRuleNOT_WORKING(Function):

    @staticmethod
    def forward(ctx, module, options, kwargs, *args):

        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.options = options

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)
            ctx.reference = options["ref_fn"](*args)

            assert isinstance(ctx.reference, tuple)

            ctx.save_for_backward(*args)

        return outputs

    @staticmethod
    @torch.no_grad()  # necessary?
    def compute_relevance(
        jacobian, inputs, reference, grad_mask, modifier=lambda x: x, dim=1
    ):

        assert dim == 1 or dim == 0

        inputs = inputs - reference

        jacobian = jacobian * inputs.unsqueeze_(dim)
        modifier(jacobian)  # in-place modification

        if dim == 1:
            jacobian.mul_(grad_mask[None, :])
        else:
            jacobian.mul_(grad_mask[:, None])

        return jacobian.sum(dim)

    @staticmethod
    @torch.no_grad()  # necessary?
    def compute_output(A, B, factor):

        out = B * A.unsqueeze(1)
        out_pos = torch.clamp(out, min=0.0)
        out_neg = torch.clamp(out, max=0.0)

        output = out.sum(0)
        output_pos = out_pos.sum(0)
        output_neg = out_neg.sum(0)
        max_abs = abs(out).max(0)[0]

        gamma = max_abs / factor - output
        gamma = gamma / (
            (output > 0) * output_pos
            + (output < 0) * output_neg
            + (output == 0) * 1e-10
        )

        return (
            output + (output > 0) * output_pos * gamma + (output < 0) * output_neg,
            gamma,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        # -- initialize variables
        A, B = ctx.saved_tensors[:2]
        A_shape, B_shape = A.shape, B.shape
        A_ref, B_ref = ctx.reference

        if ctx.options["transpose"]:
            B = B.transpose(-2, -1)
            d_k = A.shape[-1]

        # reshape head dimension into batch dimension for easier handling
        A = A.reshape(-1, *A.shape[2:])
        B = B.reshape(-1, *B.shape[2:])
        grad_outputs = grad_outputs[0].reshape(-1, *grad_outputs[0].shape[2:])

        # -- compute output
        output, gamma = vmap(
            vmap(
                IRefXGMultiplyAdaptiveGammaVMAPRule.compute_output,
                in_dims=(0, None, None),
            ),
            in_dims=(0, 0, None),
            chunk_size=ctx.options["chunk_size"],
        )(A, B, ctx.options["factor"])

        if ctx.options["transpose"]:
            output = output / math.sqrt(d_k)

        # -- compute relevance
        grad_outputs.div_(stabilize(output))

        relevance_A = vmap(
            vmap(IRefXGMultiplyVMAPRule.compute_relevance, in_dims=(None, 0, None, 0)),
            in_dims=(0, 0, None, 0),
            chunk_size=ctx.options["chunk_size"],
        )(B, A, A_ref, grad_outputs, modifier=modifier, dim=1)

        relevance_B = vmap(
            vmap(
                IRefXGMultiplyVMAPRule.compute_relevance,
                in_dims=(None, 1, None, 1),
                out_dims=1,
            ),
            in_dims=(0, 0, None, 0),
            chunk_size=ctx.options["chunk_size"],
        )(A, B, B_ref, grad_outputs, modifier=modifier, dim=0)

        if ctx.options["transpose"]:
            relevance_B = relevance_B.transpose(-2, -1)
            relevance_B = relevance_B / math.sqrt(d_k)
            relevance_A = relevance_A / math.sqrt(d_k)

        return (None, None, None) + (
            relevance_A.view(*A_shape),
            relevance_B.view(*B_shape),
        )


class IRefXGJacobianGenericRuleVMAP(Function):

    @staticmethod
    def forward(ctx, module, modifier, root_fn, kwargs, *args):
        ctx.save_for_backward(*args)
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.modifier = modifier

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)
            ctx.root = root_fn(*args)

            assert isinstance(ctx.root, tuple)

        return outputs

    @staticmethod
    def modified_output(J, inputs, grad_output, modifier):
        """
        outputs = sum_i mod(J_ji * inputs_i) = mod(J * inputs) @ ones
        """

        batch_size = len(inputs)
        expanded_dim = [1] * len(grad_output.shape[1:])

        mod_hadamard = modifier(
            J * inputs.view(batch_size, *expanded_dim, *inputs.shape[1:])
        )

        ones = torch.ones_like(inputs).view(batch_size, -1)
        mod_hadamard = mod_hadamard.view(batch_size, *grad_output.shape[1:], -1)

        return torch.einsum("b...i,bi->b...", mod_hadamard, ones)

    @staticmethod
    def modified_relevance(J, inputs, grad_mask, modifier):
        """
        relevance = sum_j mod(J_ji * inputs_i) * R_j/outputs = mod(J * inputs).T @ R/outputs
        with grad_mask = R/outputs
        """

        batch_size = len(inputs)
        expanded_dim = [1] * len(grad_mask.shape[1:])

        mod_hadamard = modifier(
            J * inputs.view(batch_size, *expanded_dim, *inputs.shape[1:])
        )

        grad_mask = grad_mask.view(batch_size, -1)
        mod_hadamard = mod_hadamard.view(batch_size, -1, *inputs.shape[1:])

        return torch.einsum("bk,bk...->b...", grad_mask, mod_hadamard)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        n_inputs = len(ctx.saved_tensors)
        n_outputs = len(grad_outputs)
        inputs_r = tuple(ctx.saved_tensors[i] - ctx.root[i] for i in range(n_inputs))

        def myfunc(*args):
            return ctx.module(*args, **ctx.saved_kwargs)

        # linearize the function at the actual input point
        # jacobian tensor structure: (*output_shape, *input_shape)
        jacobian = vmap(jacrev(myfunc, argnums=tuple(n for n in range(n_inputs))))(
            *ctx.saved_tensors
        )

        # convert jacobian to consistent data structure for easier handling
        # jacobian tuple structure: (n_outputs, n_inputs)
        if n_outputs == 1:
            jacobian = (jacobian,)

        # compute outputs as jacobian vector product: jacobian @ (inputs-root)
        outputs = [torch.zeros_like(grad_outputs[k]) for k in range(n_outputs)]
        for k in range(n_outputs):
            for i in range(n_inputs):

                output = IRefXGJacobianGenericRule.modified_output(
                    jacobian[k][i], inputs_r[i], grad_outputs[k], ctx.modifier
                )
                outputs[k] += output

        # compute relevance / outputs (Rout.T/out.T)
        grad_masks = tuple(
            grad_outputs[k] / stabilize(outputs[k]) for k in range(len(outputs))
        )

        # compute relevance as vector jacobian product: (input.T-root.T) * (Rout.T/out.T @ jacobian)
        relevances = [torch.zeros_like(ctx.saved_tensors[i]) for i in range(n_inputs)]
        for i in range(n_inputs):
            for k in range(n_outputs):

                relevance = IRefXGJacobianGenericRule.modified_relevance(
                    jacobian[k][i], inputs_r[i], grad_masks[k], ctx.modifier
                )
                relevances[i] += relevance

        return (None, None, None, None) + tuple(relevances)


class IRefXGSoftmaxJacobianGenericRule(Function):

    @staticmethod
    def forward(ctx, module, modifier, root_fn, kwargs, *args):
        ctx.save_for_backward(*args)
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.modifier = modifier

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)
            ctx.root = root_fn(*args)

            assert isinstance(ctx.root, tuple)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        input_shape = ctx.saved_tensors[0].shape
        inputs_r = ctx.saved_tensors[0] - ctx.root[0]

        def myfunc(*args):
            return ctx.module(*args, **ctx.saved_kwargs)

        # linearize the function at the actual input point
        inputs_r = inputs_r.view(-1, inputs_r.shape[-1])
        # jacobian tensor structure: (batchsize, inputs_r.shape[-1] (output), inputs_r.shape[-1]  (input))
        jacobian = vmap(jacrev(myfunc), in_dims=0)(inputs_r)

        # compute outputs as jacobian vector product: jacobian @ (inputs-root)

        mod_jac = ctx.modifier(jacobian * inputs_r.unsqueeze(1))
        output = torch.einsum("bik,bk->bi", mod_jac, torch.ones_like(inputs_r))

        # compute relevance / outputs (Rout.T/out.T)
        grad_masks = grad_outputs[0].view(-1, grad_outputs[0].shape[-1]) / stabilize(
            output
        )

        # compute relevance as vector jacobian product: (input.T-root.T) * (Rout.T/out.T @ jacobian)
        relevance = torch.einsum("bik,bk->bi", mod_jac.transpose(1, 2), grad_masks)

        return (None, None, None, None) + (relevance.view(*input_shape),)


class IRefXGSoftmaxEpsilonRule(Function):

    @staticmethod
    def forward(ctx, module, root_fn, kwargs, *args):
        ctx.saved_kwargs = kwargs
        ctx.module = module

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)

            ctx.save_for_backward(*args, outputs)
            ctx.root = root_fn(*args)

            assert isinstance(ctx.root, tuple)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        input_shape = ctx.saved_tensors[0].shape
        inputs_r = ctx.saved_tensors[0] - ctx.root[0]

        # reshape for easier handling to [batch_dimension, features]
        inputs = ctx.saved_tensors[0].view(-1, ctx.saved_tensors[0].shape[-1])
        inputs_r = inputs_r.view(*inputs.shape)
        S = ctx.saved_tensors[1].view(*inputs.shape)
        grad_outputs = grad_outputs[0].view(-1, grad_outputs[0].shape[-1])

        # --- compute derivative of softmax for obtaining the linearized output
        # (it depends on i==j or i!=j in the last dimension, so we have to compute two variants)

        # derivative of softmax for i==j
        S_ii = S * (1 - S)
        # output for case i==j
        output_ii = S_ii * inputs_r

        # type of kronecker delta to care for different derivatives of softmax depending on i==j or i!=j
        delta = torch.ones(S.shape[-1], S.shape[-1]).to(S.device)
        delta.fill_diagonal_(0)

        # output for case i!=j
        output_ji = torch.einsum("bi, bj, bi, ij -> bj", -S, S, inputs_r, delta)
        output = output_ji + output_ii

        # --- compute relevance
        grad_mask = grad_outputs / stabilize(output)
        # case i!=j
        relevance = torch.einsum(
            "bi, bj, bi, ij, bj -> bi", -S, S, inputs_r, delta, grad_mask
        )
        # case i==j
        relevance += output_ii * grad_mask

        return (None, None, None) + (relevance.view(*input_shape),)


class IRefXGSoftmaxEpsilonAbsOutRule(Function):

    @staticmethod
    def forward(ctx, module, root_fn, kwargs, *args):
        ctx.saved_kwargs = kwargs
        ctx.module = module

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)

            ctx.save_for_backward(*args, outputs)
            ctx.root = root_fn(*args)

            assert isinstance(ctx.root, tuple)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        input_shape = ctx.saved_tensors[0].shape
        inputs_r = ctx.saved_tensors[0] - ctx.root[0]

        # reshape for easier handling to [batch_dimension, features]
        inputs = ctx.saved_tensors[0].view(-1, ctx.saved_tensors[0].shape[-1])
        inputs_r = inputs_r.view(*inputs.shape)
        S = ctx.saved_tensors[1].view(*inputs.shape)
        grad_outputs = grad_outputs[0].view(-1, grad_outputs[0].shape[-1])

        # --- compute derivative of softmax for obtaining the linearized output
        # (it depends on i==j or i!=j in the last dimension, so we have to compute two variants)

        # derivative of softmax for i==j
        S_ii = S * (1 - S)
        # output for case i==j
        output_ii = S_ii * inputs_r

        # type of kronecker delta to care for different derivatives of softmax depending on i==j or i!=j
        delta = torch.ones(S.shape[-1], S.shape[-1]).to(S.device)
        delta.fill_diagonal_(0)

        # output for case i!=j
        output_ji = torch.einsum("bi, bj, bi, ij -> bj", -S, S, inputs_r, delta)
        output = output_ji + output_ii

        # --- compute relevance
        grad_mask = grad_outputs / stabilize(abs(output))
        # case i!=j
        relevance = torch.einsum(
            "bi, bj, bi, ij, bj -> bi", -S, S, inputs_r, delta, grad_mask
        )
        # case i==j
        relevance += output_ii * grad_mask

        return (None, None, None) + (relevance.view(*input_shape),)


class SoftmaxValueDTApproxRule(Function):

    @staticmethod
    def forward(ctx, module, options, kwargs, *args):
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.options = options

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)

            ctx.save_for_backward(*args, outputs)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        x, v, _ = ctx.saved_tensors
        x_shape = x.shape
        v_shape = v.shape
        x = x.view(-1, *x.shape[2:])
        v = v.reshape(-1, *v.shape[2:])
        grad_outputs = grad_outputs[0].reshape(-1, *grad_outputs[0].shape[2:])
        N = x.shape[-1]
        # S = x.softmax(-1)

        # --- compute relevance
        x = ctx.options["modifier"](x)
        lin_output = (
            torch.einsum("bjl, bkj -> bkl", v, x) * 1 / N + v.sum(1)[:, None, :] * 1 / N
        )

        out_relevance = grad_outputs / stabilize(lin_output)

        relevance_x = torch.einsum("bjl, bkl -> bkj", v, out_relevance) * 1 / N * x
        relevance_v = 1 / N * v * out_relevance.sum(1)[:, None, :]

        return (None, None, None) + (
            relevance_x.view(*x_shape),
            relevance_v.view(*v_shape),
        )


class SoftmaxValueIxGVMAPRule(Function):

    @staticmethod
    def forward(ctx, module, options, kwargs, *args):
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.options = options

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)

            ctx.save_for_backward(*args, outputs)

        return outputs

    @staticmethod
    @torch.no_grad()
    def compute_relevance(
        S, x, v, grad_outputs, output, modifier=None, virtual_bias=None
    ):

        jacobian = -S[:, :, None] * S[:, None, :]  # for j!=i
        jacobian.diagonal(dim1=1, dim2=2).copy_(S * (1 - S))  # for j==i

        jacobian.multiply_(x[:, None, :])
        modifier(jacobian)

        if not virtual_bias:
            output = torch.einsum("kji, jl -> kl", jacobian, v) + torch.einsum(
                "kj, jl -> kl", S, v
            )

        relevance_z = grad_outputs / output

        relevance_x = torch.einsum("kji, jl, kl -> ki", jacobian, v, relevance_z)
        relevance_v = torch.einsum("kj, jl, kl -> jl", S, v, relevance_z)

        return relevance_x, relevance_v

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        x, v, output = ctx.saved_tensors
        x_shape = x.shape
        v_shape = v.shape
        x = x.view(-1, *x.shape[2:])
        v = v.reshape(-1, *v.shape[2:])
        output = output.reshape(-1, *output.shape[2:])
        grad_outputs = grad_outputs[0].reshape(-1, *grad_outputs[0].shape[2:])
        S = x.softmax(-1)

        # --- compute relevance
        relevance_x, relevance_v = vmap(
            SoftmaxValueIxGVMAPRule.compute_relevance,
            chunk_size=ctx.options["chunk_size"],
            in_dims=(0, 0, 0, 0, 0),
        )(
            S,
            x,
            v,
            grad_outputs,
            output,
            modifier=ctx.options["modifier"],
            virtual_bias=ctx.options["virtual_bias"],
        )

        return (None, None, None) + (
            relevance_x.view(*x_shape),
            relevance_v.view(*v_shape),
        )


class SoftmaxDTPassRule(Function):

    @staticmethod
    def forward(ctx, module, options, kwargs, *args):
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.options = options

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)

            ctx.save_for_backward(*args, outputs)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        inputs = ctx.saved_tensors[0]
        N = inputs.shape[-1]
        S = ctx.saved_tensors[1]

        # --- compute relevance
        inputs = ctx.options["modifier"](inputs)

        if ctx.options["virtual_bias"]:
            outputs = S
        elif ctx.options["abs"]:
            outputs = 1 / N + 1 / N * inputs.abs()
        else:
            outputs = 1 / N + 1 / N * inputs

        relevance = 1 / N * inputs * grad_outputs[0] / stabilize(outputs)

        return (None, None, None) + (relevance,)


class SoftmaxIxGPassRule(Function):

    @staticmethod
    def forward(ctx, module, options, kwargs, *args):
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.options = options

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)

            ctx.save_for_backward(*args, outputs)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        inputs = ctx.saved_tensors[0]
        N = inputs.shape[-1]
        S = ctx.saved_tensors[1]

        # derivative of softmax for i==j
        S_ii = S * (1 - S)

        # --- compute relevance
        jac_inputs = S_ii * inputs
        jac_inputs = ctx.options["modifier"](jac_inputs)

        if ctx.options["virtual_bias"]:
            outputs = S
        else:
            outputs = 1 / N + jac_inputs

        relevance = jac_inputs * grad_outputs[0] / stabilize(outputs)

        return (None, None, None) + (relevance,)


class SoftmaxStablePassRule(Function):

    @staticmethod
    def forward(ctx, module, options, kwargs, *args):
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.options = options

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)

            ctx.save_for_backward(*args, outputs)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        inputs = ctx.saved_tensors[0]

        relevance = inputs * grad_outputs[0] / stabilize(inputs)

        return (None, None, None) + (relevance,)


class SoftmaxValueDTPassRule(Function):

    @staticmethod
    def forward(ctx, module, options, kwargs, *args):
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.options = options

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)

            ctx.save_for_backward(*args, outputs)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        x, v, S = ctx.saved_tensors

        # --- compute relevance
        x = ctx.options["modifier"](x)

        if ctx.options["virtual_bias"]:
            outputs = S
        else:
            outputs = v * x + v

        relevance_x = x * v * grad_outputs[0] / stabilize(outputs)
        relevance_v = v * grad_outputs[0] / stabilize(outputs)

        return (None, None, None) + (relevance_x, relevance_v)


class SoftmaxDTConditionalPassRule(Function):

    @staticmethod
    def forward(ctx, module, options, kwargs, *args):
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.options = options

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)

            ctx.save_for_backward(*args, outputs)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        inputs = ctx.saved_tensors[0]
        N = inputs.shape[-1]

        # --- compute relevance
        outputs = 1 / N + 1 / N * inputs
        inputs = (outputs < 0) * 0 + (outputs >= 0) * inputs

        relevance = 1 / N * inputs * grad_outputs[0] / stabilize(outputs)

        return (None, None, None) + (relevance,)


class IxGSoftmaxBriefRule(Function):

    @staticmethod
    def forward(ctx, module, kwargs, *args):
        ctx.saved_kwargs = kwargs
        ctx.module = module

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)

            ctx.save_for_backward(*args, outputs)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        inputs = ctx.saved_tensors[0]
        S = ctx.saved_tensors[1]

        # --- compute relevance
        relevance = inputs * (
            grad_outputs[0] - S * grad_outputs[0].sum(-1, keepdim=True)
        )

        return (None, None) + (relevance,)


class IxGSoftmaxBriefMeanRule(Function):

    @staticmethod
    def forward(ctx, module, kwargs, *args):
        ctx.saved_kwargs = kwargs
        ctx.module = module

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)

            ctx.save_for_backward(*args, outputs)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        inputs = ctx.saved_tensors[0]
        S = ctx.saved_tensors[1]

        # --- compute relevance
        relevance = inputs * (
            grad_outputs[0] - S * grad_outputs[0].sum(-1, keepdim=True)
        )

        # relevance = relevance.mean(1, keepdim=True).repeat(1, inputs.shape[1], 1, 1)
        # discard_th = 0.1
        # discard = torch.quantile(relevance, discard_th)
        # relevance[relevance < discard] = 0.0

        return (None, None) + (relevance,)


class IRefXGSoftmaxGenericRule(Function):

    @staticmethod
    def forward(ctx, module, modifiers, root_fn, kwargs, *args):
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.modifiers = modifiers

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)

            ctx.save_for_backward(*args, outputs)
            ctx.root = root_fn(*args)

            assert isinstance(ctx.root, tuple)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        input_modifiers, jac_modifiers = ctx.modifiers[:2]
        output_modifier, gradient_modifier = ctx.modifiers[2:4]
        relevance_modifier = ctx.modifiers[4]

        input_shape = ctx.saved_tensors[0].shape
        inputs_r = ctx.saved_tensors[0] - ctx.root[0]

        # reshape for easier handling to [batch_dimension, features]
        inputs = ctx.saved_tensors[0].view(-1, ctx.saved_tensors[0].shape[-1])
        inputs_r = inputs_r.view(*inputs.shape)
        S = ctx.saved_tensors[1].view(*inputs.shape)
        grad_outputs = grad_outputs[0].view(-1, grad_outputs[0].shape[-1])

        # --- compute derivative of softmax for obtaining the linearized output
        # (it depends on i==j or i!=j in the last dimension, so we have to compute two variants)

        # derivative of softmax for i==j
        S_ii = S * (1 - S)

        output_list = []
        for inp_mod, jac_mod in zip(input_modifiers, jac_modifiers):
            # output for case i==j
            output_ii = jac_mod(S_ii) * inp_mod(inputs_r)

            # type of kronecker delta to care for different derivatives of softmax depending on i==j or i!=j
            delta = torch.ones(S.shape[-1], S.shape[-1]).to(S.device)
            delta.fill_diagonal_(0)

            # output for case i!=j
            # S is always >= 0, needs no jac_mod
            output_ji = torch.einsum(
                "bi, bj, bi, ij -> bj", jac_mod(-S), S, inp_mod(inputs_r), delta
            )
            output = output_ji + output_ii
            output_list.append(output)

        output = output_modifier(output_list)

        # --- compute relevance
        grad_masks = gradient_modifier(
            grad_outputs, output
        )  # grad_outputs / stabilize(output)

        relevance_list = []
        for inp_mod, jac_mod, g_mask in zip(input_modifiers, jac_modifiers, grad_masks):
            # case i!=j
            relevance = torch.einsum(
                "bi, bj, bi, ij, bj -> bi",
                jac_mod(-S),
                S,
                inp_mod(inputs_r),
                delta,
                g_mask,
            )
            # case i==j
            relevance += jac_mod(S_ii) * inp_mod(inputs_r) * g_mask

            relevance_list.append(relevance)

        relevance = relevance_modifier(relevance_list)

        return (None, None, None, None) + (relevance.view(*input_shape),)


class IRefXGSoftmaxRuleVMAP(Function):

    @staticmethod
    def forward(ctx, module, options, kwargs, *args):

        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.options = options

        with torch.no_grad():  # seems not to be necessary

            # Convert args to float32
            ctx.output_dtype = args[0].dtype
            args = [arg.float() for arg in args]
            outputs = module(*args, **kwargs)
            ctx.reference = options["ref_fn"](*args)

            assert isinstance(ctx.reference, tuple)

            ctx.save_for_backward(*args, outputs.detach().clone())

        return outputs

    @staticmethod
    @torch.no_grad()  # ???
    def compute_relevance(
        S, inputs_ref, out_relevance, modifier=lambda x: x, virtual_bias=True
    ):

        # derivative of softmax
        jacobian = -S[:, None] * S[None, :]  # for i!=j
        jacobian.diagonal().copy_(S * (1 - S))  # for i==j

        jacobian = jacobian * inputs_ref[:, None]
        # jacobian.mul_(inputs_ref[:, None])

        if virtual_bias:
            # -- modified output = modified bias + modified output_lin
            output_lin = jacobian.sum(0)  # linear output
            bias = S - output_lin  # bias = outputs - output_lin
            modifier(bias)  # in-place modification

            modifier(jacobian)  # in-place modification
            output_lin.copy_(jacobian.sum(0))  # modified linear output
            output_lin.add_(bias)
        else:
            modifier(jacobian)
            output_lin = jacobian.sum(0)

        # out_relevance.div_(stabilize(output_lin))
        out_relevance = out_relevance / stabilize(output_lin)

        # compute relevance
        jacobian.mul_(out_relevance[None, :])
        return jacobian.sum(1)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        # -- initialize variables
        modifier = ctx.options["modifier"]
        inputs_ref, S = ctx.saved_tensors
        inputs_ref = inputs_ref.clone()
        S = S.clone()
        input_shape = inputs_ref.shape

        inputs_ref.sub_(ctx.reference[0])

        # reshape for easier handling to [batch_dimension, features]
        inputs_ref = inputs_ref.view(-1, inputs_ref.shape[-1])
        S = S.view(*inputs_ref.shape)
        grad_outputs = grad_outputs[0].view(-1, grad_outputs[0].shape[-1])

        if torch.isnan(grad_outputs).any():
            raise ValueError("stop here")

        # --  compute relevance
        relevance = vmap(
            IRefXGSoftmaxRuleVMAP.compute_relevance,
            chunk_size=ctx.options["chunk_size"],
        )(
            S,
            inputs_ref,
            grad_outputs,
            modifier=modifier,
            virtual_bias=ctx.options["virtual_bias"],
        )

        # print("rule ratio in/out", relevance.sum() / grad_outputs[0].sum())

        if torch.isnan(grad_outputs).any():
            raise ValueError("stop here")

        if torch.isnan(relevance).any():
            raise ValueError("stop here")

        return (None, None, None) + (relevance.view(*input_shape),)


class DTPassSoftmaxValueRule(Function):

    @staticmethod
    def forward(ctx, module, options, kwargs, *args):

        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.options = options

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)

            ctx.save_for_backward(*args, outputs.detach().clone())

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        # -- initialize variables
        modifier = ctx.options["modifier"]
        x, v, out = ctx.saved_tensors

        x_shape = x.shape
        v_shape = v.shape
        N = x.shape[-1]

        # reshape for easier handling to [batch_dimension, features]
        x = x.view(-1, x.shape[-1])
        x = modifier(x)
        v = v.reshape(-1, *v.shape[-2:])
        grad_outputs = grad_outputs[0].reshape(-1, grad_outputs[0].shape[-1])
        # output = output.view(-1, output.shape[-1])

        lin_output = 1 / N * x.sum(-1)[:, None] * v.sum(-2) + 1 / N * v.sum(-2)
        out_relevance_z = grad_outputs / lin_output

        # --  compute relevance
        relevance_x = (
            1 / N * x * v.sum(-1).sum(-1)[:, None] * out_relevance_z.sum(-1)[:, None]
        )
        relevance_v = 1 / N * v * out_relevance_z[:, None, :]

        return (None, None, None) + (
            relevance_x.view(*x_shape),
            relevance_v.view(*v_shape),
        )


class IRefXGSoftmaxInnerRuleVMAP(Function):

    @staticmethod
    def forward(ctx, module, options, kwargs, *args):

        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.options = options

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)
            ctx.reference = options["ref_fn"](*args)

            assert isinstance(ctx.reference, tuple)

            ctx.save_for_backward(*args, outputs.detach().clone())

        return outputs

    @staticmethod
    @torch.no_grad()  # ???
    def compute_relevance(
        S,
        inputs_ref,
        out_relevance,
        jac_modifier=lambda x: x,
        out_modifier=lambda x: x,
        virtual_bias=True,
    ):

        # derivative of softmax
        jacobian = -S[:, None] * S[None, :]  # for i!=j
        jacobian.diagonal().copy_(S * (1 - S))  # for i==j

        jacobian.mul_(inputs_ref[:, None])

        if virtual_bias:
            # -- modified output = modified bias + modified output_lin
            output_lin = jacobian.sum(0)  # linear output
            bias = S - output_lin  # bias = outputs - output_lin
            out_modifier(bias)  # in-place modification

            jacobian_out = jacobian.clone()
            out_modifier(jacobian_out)  # in-place modification
            output_lin.copy_(jacobian_out.sum(0))  # modified linear output
            output_lin.add_(bias)

            jac_modifier(jacobian)

        else:
            jacobian_out = jacobian.clone()
            out_modifier(jacobian_out)  # in-place modification
            output_lin = jacobian_out.sum(0)

            jac_modifier(jacobian)

        out_relevance.div_(stabilize(abs(output_lin)))

        # compute relevance
        jacobian.mul_(out_relevance[None, :])
        return jacobian.sum(1)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        # -- initialize variables
        jac_modifier, out_modifier = (
            ctx.options["jac_modifier"],
            ctx.options["out_modifier"],
        )
        inputs_ref, S = ctx.saved_tensors
        input_shape = inputs_ref.shape

        inputs_ref.sub_(ctx.reference[0])

        # reshape for easier handling to [batch_dimension, features]
        inputs_ref = inputs_ref.view(-1, inputs_ref.shape[-1])
        S = S.view(*inputs_ref.shape)
        grad_outputs = grad_outputs[0].view(-1, grad_outputs[0].shape[-1])

        # --  compute relevance
        relevance = vmap(
            IRefXGSoftmaxInnerRuleVMAP.compute_relevance,
            chunk_size=ctx.options["chunk_size"],
        )(
            S,
            inputs_ref,
            grad_outputs,
            jac_modifier=jac_modifier,
            out_modifier=out_modifier,
            virtual_bias=ctx.options["virtual_bias"],
        )

        return (None, None, None) + (relevance.view(*input_shape),)


class IRefXGMultiplyGenericRule(Function):

    @staticmethod
    def forward(ctx, module, modifiers, root_fn, transpose, kwargs, *args):
        ctx.save_for_backward(*args)
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.modifiers = modifiers
        ctx.transpose = transpose

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)
            ctx.root = root_fn(*args)

            assert isinstance(ctx.root, tuple)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        input_modifiers, jac_modifiers = ctx.modifiers[:2]
        output_modifier, gradient_modifier = ctx.modifiers[2:4]
        relevance_modifier = ctx.modifiers[4]

        A, B = ctx.saved_tensors
        A_root, B_root = ctx.root
        A_shape = A.shape
        B_shape = B.shape

        if ctx.transpose:
            B = B.transpose(-2, -1)

        # reshape head dimension into batch dimension for easier handling
        A = A.reshape(-1, *A.shape[2:])
        B = B.reshape(-1, *B.shape[2:])
        A_ref = A - A_root.reshape(-1, *A_root.shape[2:])
        B_ref = B - B_root.reshape(-1, *B_root.shape[2:])
        grad_outputs = grad_outputs[0].reshape(-1, *grad_outputs[0].shape[2:])

        # compute linearized output
        output_list = []
        for inp_mod, jac_mod in zip(input_modifiers, jac_modifiers):
            out = torch.einsum(
                "bil,bki->bkl", jac_mod(B), inp_mod(A_ref)
            ) + torch.einsum("bki,bil->bkl", jac_mod(A), inp_mod(B_ref))
            output_list.append(out)

        output = output_modifier(output_list)

        # --- compute relevance
        grad_masks = gradient_modifier(grad_outputs, output)

        relevance_A, relevance_B = [], []
        for inp_mod, jac_mod, g_mask in zip(input_modifiers, jac_modifiers, grad_masks):

            relevance_A.append(
                torch.einsum("bil,bki,bkl->bki", jac_mod(B), inp_mod(A_ref), g_mask)
            )
            relevance_B.append(
                torch.einsum("bki,bil,bkl->bil", jac_mod(A), inp_mod(B_ref), g_mask)
            )

        relevance_A = relevance_modifier(relevance_A)
        relevance_B = relevance_modifier(relevance_B)

        if ctx.transpose:
            relevance_B = relevance_B.transpose(-2, -1)

        return (None, None, None, None, None) + (
            relevance_A.view(*A_shape),
            relevance_B.view(*B_shape),
        )


class IRefXGMultiplyVMAPRule(Function):

    @staticmethod
    def forward(ctx, module, options, kwargs, *args):

        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.options = options

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)
            ctx.reference = options["ref_fn"](*args)

            assert isinstance(ctx.reference, tuple)

            if options["virtual_bias"]:
                ctx.save_for_backward(*args, outputs.detach().clone())
            else:
                ctx.save_for_backward(*args)

        return outputs

    @staticmethod
    @torch.no_grad()  # necessary?
    def compute_relevance(
        jacobian, inputs, reference, grad_mask, modifier=lambda x: x, dim=1
    ):

        assert dim == 1 or dim == 0

        inputs = inputs - reference

        jacobian = jacobian * inputs.unsqueeze_(dim)
        modifier(jacobian)  # in-place modification

        if dim == 1:
            jacobian.mul_(grad_mask[None, :])
        else:
            jacobian.mul_(grad_mask[:, None])

        return jacobian.sum(dim)

    @staticmethod
    @torch.no_grad()  # necessary?
    def compute_output(A, B, A_ref, B_ref, modifier=lambda x: x):

        out_A = B * (A - A_ref).unsqueeze(1)
        modifier(out_A)

        out_B = A.unsqueeze(1) * (B - B_ref)
        modifier(out_B)

        return out_A.sum(0) + out_B.sum(0)
        # out = torch.einsum("bil,bki->bkl", jac_mod(B), inp_mod(A_ref)) + torch.einsum("bki,bil->bkl", jac_mod(A), inp_mod(B_ref))

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        # -- initialize variables
        A, B = ctx.saved_tensors[:2]
        A_shape, B_shape = A.shape, B.shape
        A_ref, B_ref = ctx.reference

        if ctx.options["virtual_bias"]:
            output = ctx.saved_tensors[2]
            output = output.reshape(-1, *output.shape[2:])

        if ctx.options["transpose"]:
            B = B.transpose(-2, -1)
            d_k = A.shape[-1]

        modifier = ctx.options["modifier"]

        # reshape head dimension into batch dimension for easier handling
        A = A.reshape(-1, *A.shape[2:])
        B = B.reshape(-1, *B.shape[2:])
        grad_outputs = grad_outputs[0].reshape(-1, *grad_outputs[0].shape[2:])

        # -- compute output
        output_mod_lin = vmap(
            vmap(IRefXGMultiplyVMAPRule.compute_output, in_dims=(0, None, None, None)),
            in_dims=(0, 0, None, None),
            chunk_size=ctx.options["chunk_size"],
        )(A, B, A_ref, B_ref, modifier=modifier)

        if ctx.options["transpose"]:
            output_mod_lin = output_mod_lin / math.sqrt(d_k)

        if ctx.options["virtual_bias"]:
            # compute modified out = modified bias + modified output_lin
            output_lin = vmap(
                vmap(
                    IRefXGMultiplyVMAPRule.compute_output, in_dims=(0, None, None, None)
                ),
                in_dims=(0, 0, None, None),
                chunk_size=ctx.options["chunk_size"],
            )(A, B, A_ref, B_ref, modifier=lambda x: x)

            if ctx.options["transpose"]:
                output_lin = output_lin / math.sqrt(d_k)

            bias = output - output_lin
            modifier(bias)
            output_lin = output_mod_lin + bias
        else:
            output_lin = output_mod_lin

        # -- compute relevance
        grad_outputs.div_(stabilize(output_lin))

        relevance_A = vmap(
            vmap(IRefXGMultiplyVMAPRule.compute_relevance, in_dims=(None, 0, None, 0)),
            in_dims=(0, 0, None, 0),
            chunk_size=ctx.options["chunk_size"],
        )(B, A, A_ref, grad_outputs, modifier=modifier, dim=1)

        relevance_B = vmap(
            vmap(
                IRefXGMultiplyVMAPRule.compute_relevance,
                in_dims=(None, 1, None, 1),
                out_dims=1,
            ),
            in_dims=(0, 0, None, 0),
            chunk_size=ctx.options["chunk_size"],
        )(A, B, B_ref, grad_outputs, modifier=modifier, dim=0)

        if ctx.options["transpose"]:
            relevance_B = relevance_B.transpose(-2, -1)
            relevance_B = relevance_B / math.sqrt(d_k)
            relevance_A = relevance_A / math.sqrt(d_k)

        return (None, None, None) + (
            relevance_A.view(*A_shape),
            relevance_B.view(*B_shape),
        )


class IRefXGClipRule(Function):

    @staticmethod
    def forward(ctx, module, root_fn, factor, kwargs, *args):
        ctx.save_for_backward(*args)
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.factor = factor

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)
            ctx.root = root_fn(*args)

            assert isinstance(ctx.root, tuple)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        inputs_r = tuple(
            ctx.saved_tensors[i] - ctx.root[i] for i in range(len(ctx.saved_tensors))
        )

        def myfunc(*args):
            return ctx.module(*args, **ctx.saved_kwargs)

        # ctx.saved_tensors and inputs_r must have same shapes
        _, Jvs = jvp(myfunc, ctx.saved_tensors, inputs_r)

        if isinstance(Jvs, tuple):  # TODO: merge
            outputs = tuple(Jvs[i] for i in range(len(Jvs)))
            grad_masks = tuple(
                grad_outputs[i] / stabilize(outputs[i]) for i in range(len(outputs))
            )
        else:
            outputs = Jvs
            grad_masks = grad_outputs[0] / stabilize(outputs)

        if not isinstance(grad_masks, tuple):
            grad_masks = (grad_masks,)

        for data in grad_masks:

            dims = tuple(range(1, data.dim()))
            batch_mask = data != 0
            num_non_zeros = batch_mask.sum(dim=dims, keepdim=True)
            batch_mean = (
                torch.sum(data * batch_mask, dim=dims, keepdim=True) / num_non_zeros
            )
            batch_std = torch.sqrt(
                torch.sum(
                    ((data - batch_mean) ** 2) * batch_mask, dim=dims, keepdim=True
                )
                / num_non_zeros
            )

            # Compute the minimum and maximum values to keep
            min_val = batch_mean - ctx.factor * batch_std
            max_val = batch_mean + ctx.factor * batch_std

            # Set values outside of 95% to the mean for each batch separately
            # expand like tensor min_val
            torch.where(data >= min_val, data, torch.full_like(data, 0), out=data)
            torch.where(data <= max_val, data, torch.full_like(data, 0), out=data)

        _, vjpfunc = vjp(myfunc, *ctx.saved_tensors)
        if len(grad_masks) > 1:
            grads = vjpfunc(grad_masks)
        else:
            grads = vjpfunc(grad_masks[0])

        return (None, None, None, None) + tuple(
            grads[i] * inputs_r[i] if grads[i] != None else None
            for i in range(len(inputs_r))
        )


class IRefXGSoftmaxValueMulRuleVMAP(Function):

    @staticmethod
    def forward(ctx, module, options, kwargs, *args):

        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.options = options

        with torch.no_grad():  # seems not to be necessary
            y, attention = module(*args, **kwargs)

            ctx.reference = options["ref_fn"](*args)
            assert isinstance(ctx.reference, tuple)

            ctx.save_for_backward(*args, attention.detach().clone())

        return y, attention

    @staticmethod
    @torch.no_grad()  # ???
    def compute_lin_output(attention, x, v, modifier=lambda x: x):
        # vmap over k
        # out vmap over batch

        # derivative of softmax
        jacobian = -attention[None, :] * attention[:, None]  # for i!=j
        jacobian.diagonal().copy_(attention * (1 - attention))  # for i==j

        # first
        tmp = modifier(attention[:, None] * v)
        out = tmp.sum(0)

        # second
        def compute_inner(jacobian, x, v):
            # vmap over l (v[:, None, l])
            # returns out.shape = [l]
            return modifier(jacobian * x[None, :] * v[:, None]).sum()

        out_inner = vmap(
            compute_inner, in_dims=(None, None, 1), out_dims=0, chunk_size=32
        )(jacobian, x, v)

        return out + out_inner

    @staticmethod
    @torch.no_grad()  # ???
    def compute_bias(attention, x, v):
        # vmap over k
        # out vmap over batch

        # derivative of softmax
        jacobian = -attention[None, :] * attention[:, None]  # for i!=j
        jacobian.diagonal().copy_(attention * (1 - attention))  # for i==j

        # second
        def compute_inner(jacobian, x, v):
            # vmap over l (v[:, None, l])
            # returns out.shape = [l]
            return (jacobian * x[None, :] * v[:, None]).sum()

        out_inner = vmap(
            compute_inner, in_dims=(None, None, 1), out_dims=0, chunk_size=32
        )(jacobian, x, v)

        return -out_inner

    @staticmethod
    @torch.no_grad()  # ???
    def compute_v_relevance(attention, v, out_relevance, modifier=lambda x: x):
        # vmap over l
        # out vmap over batch

        tmp = modifier(attention * v[None, :])
        relevance = tmp * out_relevance[:, None]

        return relevance.sum(0)

    @staticmethod
    @torch.no_grad()  # ???
    def compute_x_relevance(attention, x, v, out_relevance, modifier=lambda x: x):
        # vmap over k
        # out vmap over batch

        # derivative of softmax
        jacobian = -attention[None, :] * attention[:, None]  # for i!=j
        jacobian.diagonal().copy_(attention * (1 - attention))  # for i==j

        def compute_inner(jacobian, x, v, out_relevance):
            # vmap over i (jacobian[:, i], x[i])
            # returns out.shape = [i]
            tmp = modifier(jacobian[:, None] * x * v) * out_relevance[None, :]
            return tmp.sum()

        out_inner = vmap(
            compute_inner, in_dims=(1, 0, None, None), out_dims=0, chunk_size=32
        )(jacobian, x, v, out_relevance)
        return out_inner

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        # -- initialize variables
        modifier = ctx.options["modifier"]
        x, v, attention = ctx.saved_tensors

        x_shape, v_shape = x.shape, v.shape

        x.sub_(ctx.reference[0])

        # reshape heads into batch dimension for easier handling
        x = x.view(-1, *x.shape[-2:])
        v = v.reshape(-1, *v.shape[-2:])
        attention = attention.view(-1, *attention.shape[-2:])
        out_relevance = grad_outputs[0].reshape(-1, *grad_outputs[0].shape[-2:])

        # --  compute output
        output_lin = vmap(
            vmap(
                IRefXGSoftmaxValueMulRuleVMAP.compute_lin_output, in_dims=(0, 0, None)
            ),
            in_dims=(0, 0, 0),
            out_dims=0,
            chunk_size=ctx.options["chunk_size"],
        )(attention, x, v, modifier=modifier)

        if ctx.options["virtual_bias"]:
            # -- modified output = modified bias + modified output_lin
            bias = vmap(
                vmap(IRefXGSoftmaxValueMulRuleVMAP.compute_bias, in_dims=(0, 0, None)),
                in_dims=(0, 0, 0),
                chunk_size=ctx.options["chunk_size"],
            )(attention, x, v)

            modifier(bias)  # in-place modification
            output_lin.add_(bias)

        # --  compute relevance
        out_relevance.div_(stabilize(output_lin))

        relevance_x = vmap(
            vmap(
                IRefXGSoftmaxValueMulRuleVMAP.compute_x_relevance,
                in_dims=(0, 0, None, 0),
            ),
            in_dims=(0, 0, 0, 0),
            out_dims=0,
            chunk_size=ctx.options["chunk_size"],
        )(attention, x, v, out_relevance, modifier=modifier)

        relevance_v = vmap(
            vmap(
                IRefXGSoftmaxValueMulRuleVMAP.compute_v_relevance,
                in_dims=(None, 1, 1),
                out_dims=1,
            ),
            in_dims=(0, 0, 0),
            out_dims=0,
            chunk_size=ctx.options["chunk_size"],
        )(attention, v, out_relevance, modifier=modifier)

        return (None, None, None) + (
            relevance_x.view(*x_shape),
            relevance_v.view(*v_shape),
        )


class DeepTaylorReference(Function):

    @staticmethod
    def forward(ctx, module, ref_fn, bias, distribute_bias, clip, kwargs, *args):
        ctx.save_for_backward(*args)
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.distribute_bias = distribute_bias
        ctx.bias = bias
        ctx.clip = clip

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)
            ctx.ref = ref_fn(*args)

            assert isinstance(ctx.ref, tuple)
            ctx.outputs = outputs.detach().clone()

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        saved_tensors = tuple(
            ctx.saved_tensors[i] for i in range(len(ctx.saved_tensors))
        )

        inputs_r = tuple(
            saved_tensors[i] - ctx.ref[i] for i in range(len(saved_tensors))
        )

        def myfunc(*args):
            return ctx.module(*args, **ctx.saved_kwargs)

        # compute jacobian at inputs and multiply from right side with reference point
        # ctx.saved_tensors and inputs_r must have same shapes
        _, Jvs = jvp(myfunc, ctx.ref, inputs_r)

        if ctx.bias == "error":
            bias = ctx.outputs - Jvs
        elif ctx.bias == "reference":
            bias = myfunc(*ctx.ref)
        elif ctx.bias == "zero":
            bias = 0

        if ctx.clip:
            Jvs = Jvs.clamp(min=0)

        outputs = Jvs + bias
        normed_relevance = grad_outputs[0] / stabilize(outputs)

        # compute jacobian at inputs and multiply from left side with R/output
        _, vjpfunc = vjp(myfunc, *ctx.ref)
        grads = vjpfunc(normed_relevance)

        if torch.isnan(grads[0]).any():
            raise ValueError("NaN here")

        relevance = grads[0] * inputs_r[0]

        if ctx.distribute_bias:
            assert ctx.bias != "zero"
            relevance += (bias * normed_relevance).mean(-1, keepdim=True)

        # multiply vJ with reference point
        return (None, None, None, None, None, None, relevance)


class GradientxInputReference(Function):

    @staticmethod
    def forward(ctx, module, ref_fn, bias, distribute_bias, kwargs, *args):
        ctx.save_for_backward(*args)
        ctx.saved_kwargs = kwargs
        ctx.module = module
        ctx.distribute_bias = distribute_bias
        ctx.bias = bias

        with torch.no_grad():  # seems not to be necessary
            outputs = module(*args, **kwargs)
            ctx.ref = ref_fn(*args)

            assert isinstance(ctx.ref, tuple)
            ctx.outputs = outputs.detach().clone()

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        saved_tensors = tuple(
            ctx.saved_tensors[i] for i in range(len(ctx.saved_tensors))
        )

        inputs_r = tuple(
            saved_tensors[i] - ctx.ref[i] for i in range(len(saved_tensors))
        )

        def myfunc(*args):
            return ctx.module(*args, **ctx.saved_kwargs)

        # compute jacobian at inputs and multiply from right side with reference point
        # ctx.saved_tensors and inputs_r must have same shapes
        _, Jvs = jvp(myfunc, saved_tensors, inputs_r)

        if ctx.bias == "error":
            bias = ctx.outputs - Jvs
        elif ctx.bias == "reference":
            bias = myfunc(*ctx.ref)
        elif ctx.bias == "zero":
            bias = 0

        # print("bias", abs(bias).sum().item(), "Jv", abs(Jvs).sum().item())

        outputs = Jvs + bias
        normed_relevance = grad_outputs[0] / stabilize(outputs)

        # compute jacobian at inputs and multiply from left side with R/output
        _, vjpfunc = vjp(myfunc, *saved_tensors)
        grads = vjpfunc(normed_relevance)

        if torch.isnan(grads[0]).any():
            raise ValueError("NaN here")

        relevance = grads[0] * inputs_r[0]

        if ctx.distribute_bias:
            assert ctx.bias != "zero"
            relevance += (bias * normed_relevance).mean(-1, keepdim=True)

        # multiply vJ with reference point
        return (None, None, None, None, None, relevance)


class ElementwiseMultiplyCPRule(Function):

    @staticmethod
    def forward(ctx, module, kwargs, *args):

        with torch.no_grad():
            outputs = module(*args, **kwargs)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        return None, None, None, grad_outputs[0]


class LinearEpsilonQuantizeRule(Function):

    @staticmethod
    def forward(ctx, module, kwargs, *args):
        ctx.module = module

        with torch.no_grad():
            outputs = module(*args, **kwargs)
            ctx.save_for_backward(*args, outputs)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):

        def custom_erf(x):
            return x / torch.sqrt(1 + 300 * x**2)

        def clip_relevance(x):
            return (x.abs() > 1e-5) * x

        inputs, outputs = ctx.saved_tensors
        weight = ctx.module.weight
        out_relevance = grad_outputs[0]

        # set all values smaller than 1e-3 to zero in out_relevance
        out_relevance = (out_relevance.abs() > 1e-6) * out_relevance

        out_relevance = out_relevance / stabilize(outputs)

        relevance = torch.matmul(out_relevance, custom_erf(weight)).mul_(inputs)
        # relevance = torch.matmul(out_relevance, weight).mul_(inputs)

        return (None, None, relevance)


class SumZPlus(Function):

    @staticmethod
    def forward(ctx, module, kwargs, *args):

        with torch.no_grad():
            outputs = module(*args, **kwargs)
            ctx.save_for_backward(*args)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):

        pydevd.settrace(suspend=False, trace_only_current_thread=True)

        inputs = ctx.saved_tensors

        out_relevance = grad_outputs[0]

        output_abs = abs(inputs[0] + inputs[1])
        norm_relevance = out_relevance / stabilize(output_abs)
        input_1_rel = norm_relevance * inputs[0].clamp(min=0)
        input_2_rel = norm_relevance * inputs[1].clamp(min=0)

        # output_zplus = inputs[0].clamp(min=0) + inputs[1].clamp(min=0)
        # norm_relevance = out_relevance / stabilize(output_zplus)
        # input_1_rel = norm_relevance * inputs[0].clamp(min=0)
        # input_2_rel = norm_relevance * inputs[1].clamp(min=0)

        # output_zplus = inputs[0] + inputs[1]
        # norm_relevance = out_relevance / stabilize(output_zplus)
        # input_1_rel = norm_relevance * inputs[0]
        # input_2_rel = norm_relevance * inputs[1]

        return None, None, input_1_rel, input_2_rel
