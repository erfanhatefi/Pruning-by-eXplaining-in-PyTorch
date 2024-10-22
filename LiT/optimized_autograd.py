import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn

def stabilize(input, epsilon=1e-6):
    return input.add_(epsilon)

##############
##### RULES
##############

class SoftmaxEpsilonAutograd(Function):

    @staticmethod
    def forward(ctx, inputs, dim):
        
        if inputs.requires_grad:
            with torch.enable_grad():
                outputs = F.softmax(inputs, dim=dim)
                ctx.save_for_backward(inputs, outputs)
            return outputs.detach()
        else:
            outputs = F.softmax(inputs, dim=dim)
            return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):

        inputs, outputs = ctx.saved_tensors
        relevance_norm = grad_outputs[0] / stabilize(outputs)
        grad, = torch.autograd.grad(outputs, inputs, relevance_norm)
        
        return (grad*inputs, None)
    

class SoftmaxEpsilon(Function):

    @staticmethod
    def forward(ctx, inputs, dim):
    
        outputs = F.softmax(inputs, dim=dim)
        ctx.save_for_backward(inputs, outputs)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):

        inputs, output = ctx.saved_tensors

        relevance = (grad_outputs[0].sub_(output.mul_(grad_outputs[0].sum(-1, keepdim=True)))).mul_(inputs)

       # if torch.isnan(relevance).any():
       #         raise ValueError("NaN encountered")
        
        return (relevance, None)


class LinearEpsilon(Function):

    epsilon = 1e-9

    @staticmethod
    def forward(ctx, inputs, weight, bias=None):
        
        if inputs.requires_grad:
            with torch.enable_grad():
                outputs = F.linear(inputs, weight, bias)
            ctx.save_for_backward(inputs, outputs)
            return outputs.detach()
        else:
            outputs = F.linear(inputs, weight, bias)
            return outputs
        
    @staticmethod
    def backward(ctx, *grad_outputs):

        inputs, outputs = ctx.saved_tensors

        relevance_norm = grad_outputs[0] / stabilize(outputs)

        relevance, = torch.autograd.grad(outputs, inputs, relevance_norm)
        relevance.mul_(inputs)
        
        return (relevance, None, None)



class ElementwiseIdentity(Function):
    """
    Distribute the relevance 100% to the input
    """

    @staticmethod
    def forward(ctx, module, input):

        outputs = module(input)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):

        #if torch.isnan(grad_outputs[0]).any():
        #        raise ValueError("NaN encountered")

        return (None,) + grad_outputs
    
class SiLUIdentity(Function):
    """
    Distribute the relevance 100% to the input
    """

    @staticmethod
    def forward(ctx, input):

        outputs = F.silu(input)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):

        #if torch.isnan(grad_outputs[0]).any():
        #        raise ValueError("NaN encountered")

        return grad_outputs
    

class LlamaRMSNormIdentity(Function):
    """
    Distribute the relevance 100% to the input
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, variance_epsilon):

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)

        return weight * hidden_states.to(input_dtype)

    @staticmethod
    def backward(ctx, *grad_outputs):

       # if torch.isnan(grad_outputs[0]).any():
       #         raise ValueError("NaN encountered")

        return grad_outputs + (None, None)
    

class ConstantMultiplyIdentity(Function):
    """
    Distribute the relevance 100% to the input
    """

    @staticmethod
    def forward(ctx, input, constant):
        return input * constant

    @staticmethod
    def backward(ctx, *grad_outputs):

       # if torch.isnan(grad_outputs[0]).any():
        #        raise ValueError("NaN encountered")

        return grad_outputs + (None,)


class ElementwiseMultiplyUniform(Function):
    """
    Distribute the relevance 100% to the input
    """

    @staticmethod
    def forward(ctx, input_a, input_b):
        return input_a * input_b

    @staticmethod
    def backward(ctx, *grad_outputs):

        relevance = grad_outputs[0].mul_(0.5)

       # if torch.isnan(relevance).any():
       #     raise ValueError("NaN encountered")
        
        return relevance, relevance
    

class MatrixMultiplicationEpsilon(Function):

    epsilon = 1e-9
    
    @staticmethod
    def forward(ctx, input_a, input_b):

        if input_a.requires_grad:
            with torch.enable_grad():
                outputs = torch.matmul(input_a, input_b)
            ctx.save_for_backward(input_a, input_b, outputs)
            return outputs.detach()
        else:
            outputs = torch.matmul(input_a, input_b)
            return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):

        input_a, input_b, outputs = ctx.saved_tensors
        out_relevance = grad_outputs[0]

        out_relevance = out_relevance.div_(stabilize(outputs * 2))

        relevance_a, relevance_b = torch.autograd.grad(outputs, (input_a, input_b), out_relevance)
        relevance_a.mul_(input_a)
        relevance_b.mul_(input_b)

        # if torch.isnan(relevance_a).any() or torch.isnan(relevance_b).any():
        #         raise ValueError("NaN encountered")
        
        return (relevance_a, relevance_b, None, None)
        

class SumEpsilonNotWorking(Function):

    epsilon = 1e-9
    
    @staticmethod
    def forward(ctx, input_a, input_b):

        if input_a.requires_grad:
            with torch.enable_grad():
                outputs = input_a + input_b
            ctx.save_for_backward(input_a, input_b, outputs)
            return outputs.detach()
        else:
            outputs = input_a + input_b
            return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):

        input_a, input_b, outputs = ctx.saved_tensors

        out_relevance_norm = grad_outputs[0].div_(outputs + SumEpsilon.epsilon)
        relevance_a, relevance_b = torch.autograd.grad(outputs, (input_a, input_b), out_relevance_norm)

        relevance_a = relevance_a.mul_(input_a)
        relevance_b = relevance_b.mul_(input_b)

        return relevance_a, relevance_b
    

class SumEpsilon(Function):

    epsilon = 1e-9
    
    @staticmethod
    def forward(ctx, input_a, input_b):
    
        outputs = input_a + input_b
        ctx.save_for_backward(input_a, input_b)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):

        input_a, input_b = ctx.saved_tensors
        #out_relevance_norm = grad_outputs[0] / (sum(args) + SumEpsilon.epsilon)

        out_relevance_norm = grad_outputs[0].div_(stabilize(input_a + input_b))
        
        relevance_a = input_a * out_relevance_norm 
        relevance_b = out_relevance_norm.mul_(input_b)

      #  if any(torch.isnan(tensor).any() for tensor in relevances):
      #      raise ValueError("NaN encountered")

        return relevance_a, relevance_b
    

########
### Modules
########


class SoftmaxEpsilonRule(nn.Module):

    def __init__(self, module) -> None:
        super().__init__()

    def forward(self, inputs):
        return SoftmaxEpsilon.apply(inputs)


class LinearEpsilonRule(nn.Module):

    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return LinearEpsilon.apply(inputs, self.module.weight, self.module.bias)
    

class ElementwiseIdentityRule(nn.Module):

    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args):
        return ElementwiseIdentity.apply(self.module, *args)
    

