import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn

def stabilize(input, epsilon=1e-6, inplace=True):
    if inplace:
        return input.add_(epsilon)
    else:
        return input.add(epsilon)

##############
##### RULES
##############

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


class LinearEpsilon2(Function):

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
        #init_relevance = grad_outputs[0]

        #relevance_norm = grad_outputs[0] / (outputs + LinearEpsilon.epsilon)
        relevance_norm = grad_outputs[0] / stabilize(outputs)
        
        #init_relevance.div_(outputs.add_(LinearEpsilon.epsilon))

        grad, = torch.autograd.grad(outputs, inputs, relevance_norm)
        #relevance = torch.einsum("bhi, ji, bhj-> bhi", inputs, weight, init_relevance)
        
        return (grad*inputs, None, None)


class LinearEpsilon(Function):

    epsilon = 1e-9

    @staticmethod
    def forward(ctx, inputs, weight, bias=None):
        
        outputs = F.linear(inputs, weight, bias)
        ctx.save_for_backward(inputs, weight, outputs)
    
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):

        inputs, weight, outputs = ctx.saved_tensors
        out_relevance = grad_outputs[0]

        # init_relevance.div_(outputs.add_(LinearEpsilon.epsilon))
        out_relevance = out_relevance / stabilize(outputs)

        relevance = torch.matmul(out_relevance, weight).mul_(inputs)
        
        return (relevance, None, None)


class LinearEpsilon2(Function):

    epsilon = 1e-9

    @staticmethod
    def forward(ctx, inputs, weight, bias=None):
        
        outputs = F.linear(inputs, weight, bias)
        ctx.save_for_backward(inputs, weight, bias)
    
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):

        inputs, weight, bias = ctx.saved_tensors
        out_relevance = grad_outputs[0]

        outputs = F.linear(inputs, weight, bias)

        # init_relevance.div_(outputs.add_(LinearEpsilon.epsilon))
        out_relevance = out_relevance / stabilize(outputs)

        relevance = torch.matmul(out_relevance, weight).mul_(inputs)
        
        return (relevance, None, None)


class ElementwiseIdentity(Function):
    """
    Distribute the relevance 100% to the input
    """

    @staticmethod
    def forward(ctx, module, input):

        with torch.no_grad(): # necessary? TODO
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

        outputs = F.silu(input, inplace=False)

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
        
        return relevance, relevance
    
class ElementwiseMultiplyCP(Function):
    """
    Distribute the relevance 100% to the input
    """

    @staticmethod
    def forward(ctx, input_a, input_b):
        return input_a * input_b

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, grad_outputs[0]


class MatrixMultiplicationEpsilon(Function):
    
    @staticmethod
    def forward(ctx, input_a, input_b):
        
        outputs = torch.matmul(input_a, input_b)
        ctx.save_for_backward(input_a, input_b, outputs)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):

        input_a, input_b, outputs = ctx.saved_tensors
        out_relevance = grad_outputs[0]

        #out_relevance = out_relevance.div_(stabilize(outputs.mul_(2)))
        #relevance_a = torch.matmul(out_relevance, input_b.permute(0, 1, -1, -2)).mul_(input_a)
        #relevance_b = torch.matmul(input_a.permute(0, 1, -1, -2), out_relevance).mul_(input_b)

        out_relevance = out_relevance.div_(stabilize(outputs * 2))

        relevance_a = torch.matmul(out_relevance, input_b.permute(0, 1, -1, -2)).mul_(input_a)
        relevance_b = torch.matmul(input_a.permute(0, 1, -1, -2), out_relevance).mul_(input_b)
        
        return (relevance_a, relevance_b)
    

class MatrixMultiplicationCP(Function):
    
    @staticmethod
    def forward(ctx, input_a, input_b):
        
        outputs = torch.matmul(input_a, input_b)
        ctx.save_for_backward(input_a, input_b, outputs)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):

        input_a, input_b, outputs = ctx.saved_tensors
        out_relevance = grad_outputs[0]

        out_relevance = out_relevance.div_(stabilize(outputs, inplace=False))
        relevance_b = torch.matmul(input_a.permute(0, 1, -1, -2), out_relevance).mul_(input_b)
        
        return (None, relevance_b)

        

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
    

class SumEpsilonFaster(Function):

    epsilon = 1e-9
    
    @staticmethod
    def forward(ctx, *args):
    
        outputs = sum(args)
        ctx.save_for_backward(*args, outputs)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):

        a, b, outputs = ctx.saved_tensors
        #out_relevance_norm = grad_outputs[0] / (sum(args) + SumEpsilon.epsilon)

        out_relevance_norm = grad_outputs[0] / stabilize(outputs)
        

        relevances = tuple(i * out_relevance_norm for i in [a, b])

      #  if any(torch.isnan(tensor).any() for tensor in relevances):
      #      raise ValueError("NaN encountered")

        return relevances
    


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
    

