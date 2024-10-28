import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.swin_transformer import ShiftedWindowAttentionV2
import numpy as np

### --- Canonize ElementwiseMultiply --- ###
class ElementwiseMultiplyLiT(nn.Module):

    def forward(self, a, b):
        return a * b
    
### --- Canonize ConstantMultiply --- ###
class ConstantMultiplyLiT(nn.Module):

    def forward(self, a, constant=1):
        return a * constant


### --- Canonize Sum --- ###
class SumLiT(nn.Module):

    def forward(self, a, b):
        if not isinstance(a, (torch.Tensor, int, float, np.ndarray)) or not isinstance(b, (torch.Tensor, int, float, np.ndarray)):
            raise TypeError("The canonization process encapsulates each addition operation into a nn.Module.\
                            However, this is not a valid tensor addition operation and should not be canonized. Please remove the sum_layer_lit at this line.")
        return a + b
    
class Sum3LiT(nn.Module):

    def forward(self, a, b, c):
        if not isinstance(a, (torch.Tensor, int, float, np.ndarray)) or not isinstance(b, (torch.Tensor, int, float, np.ndarray)):
            raise TypeError("The canonization process encapsulates each addition operation into a nn.Module.\
                            However, this is not a valid tensor addition operation and should not be canonized. Please remove the sum_layer_lit at this line.")
        return a + b + c

### --- Canonize LayerNorm --- ###
class LayerNormTU(torch.nn.LayerNorm): 

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (x - mean) / std.detach()
        if self.weight is not None:
            y *= self.weight
        if self.bias is not None:
            y += self.bias
        return y
    
class LinearNormMerge(nn.Linear):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def merge_weight(self, x, ln):
        #TODO: bug fix, linear_weight overwritten after second forward pass

        mean = x.mean(dim=-1, keepdim=True).detach()
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + ln.eps).sqrt().detach()

        linear_weight = self.weight.detach().clone()
        self.weight = nn.Parameter(self.weight * ln.weight / std)
        self.bias = nn.Parameter(self.bias + linear_weight @ ln.bias - (mean.repeat(1, 1, x.shape[-1]) / std ) @ (linear_weight * ln.weight).T)

    def forward(self, x):

        y = torch.bmm(x, self.weight.transpose(-1, -2))
        return y + self.bias
    

class LayerNormF32TU(torch.nn.LayerNorm): 

    def forward(self, x):
        orig_type = x.dtype
        x = x.to(torch.float32)
        
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (x - mean) / std.detach()
        if self.weight is not None:
            y *= self.weight
        if self.bias is not None:
            y += self.bias
            
        return y.to(orig_type)


### --- Reimplementation of the MultiheadAttention module separated into single modules for LiT --- ###    
class LinearInProjection(nn.Module):
    """
    nn.Linear reimplemented so that we can apply different zennit rules to the Attention Module
    independet of normal nn.Linear layers.
    """

    def __init__(self, in_features: int, out_features: int, bias=True):

        super().__init__()
        self.bias = bias
        self.proj_weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.proj_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.proj_bias = None

        self._init_parameters()

    def forward(self, x):
        
        return F.linear(x, self.proj_weight, self.proj_bias)

    def _init_parameters(self):

        nn.init.xavier_uniform_(self.proj_weight)
        if self.bias:
            self.proj_bias.data.fill_(0)


class LinearQueryProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        self.bias = bias
        self.proj_weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.proj_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.proj_bias = None
    def forward(self, x):
        return F.linear(x, self.proj_weight, self.proj_bias)

class LinearKeyProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        self.bias = bias
        self.proj_weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.proj_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.proj_bias = None
    def forward(self, x):
        return F.linear(x, self.proj_weight, self.proj_bias)
    
class LinearValueProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        self.bias = bias
        self.proj_weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.proj_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.proj_bias = None
    def forward(self, x):
        return F.linear(x, self.proj_weight, self.proj_bias)


class LinearOutProjection(nn.Module):
    """
    nn.Linear reimplemented so that we can apply different zennit rules to the Attention Module
    independet of normal nn.Linear layers.
    """

    def __init__(self, in_features: int, out_features: int, bias=True):

        super().__init__()
        self.bias = bias
        self.proj_weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.proj_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.proj_bias = None

        self._init_parameters()

    def forward(self, x):
        
        return F.linear(x, self.proj_weight, self.proj_bias)

    def _init_parameters(self):

        nn.init.xavier_uniform_(self.proj_weight)
        if self.bias:
            self.proj_bias.data.fill_(0)


class QueryKeyMultiplicationOLD(nn.Module):
    
    def forward(self, q, k, mask=None):
        d_k = q.shape[-1]
        # matmul computes matrix product on last two dimensions
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) 
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask:
            mask = self.prepare_mask(mask) 
            attn_logits = attn_logits * mask.detach()

        return attn_logits
    
    @torch.no_grad()
    def prepare_mask(self, mask):
        # -- broadcast mask
        assert mask.ndim > 2 # [..., SeqLen, SeqLen]
        if mask.ndim == 3:
            # head dimension missing since first dimension is batch
            mask = mask.unsqueeze(1)

        # -- set all values == 0 to a large negative value and the remainder to 1
        bool_mask = (mask == 0)
        bool_mask = ~bool_mask + -9e15 * bool_mask
        return bool_mask


class QueryKeyMultiplication(nn.Module):
    
    def forward(self, q, k, key_padding_mask=None, attn_mask=None, scale=None):
        
        if scale is None:
            d_k = q.shape[-1]
            scale = 1 / math.sqrt(d_k)

        # matmul computes matrix product on last two dimensions
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) 
        attn_logits = attn_logits * scale

        mask = torch.zeros_like(attn_logits, device=attn_logits.device)
        if key_padding_mask is not None:
            mask += self.prepare_key_padding_mask(key_padding_mask, attn_mask, q)
        if attn_mask is not None:
            mask += self.prepare_attn_mask(attn_mask, q)

        attn_logits = attn_logits + mask.detach()
        return attn_logits


    @torch.no_grad()
    def prepare_key_padding_mask(self, mask, attn_mask, query):
        # -- broadcast mask
        assert mask.ndim > 1 # [..., SeqLen]
        if mask.ndim == 2: # [Batch, ... , ... , SeqLen]
            b, k_len = mask.shape
            mask = mask.view(b, 1, 1, k_len)

        return F._canonical_mask(mask, "key_padding_mask", F._none_or_dtype(attn_mask), "attn_mask", query.dtype)

    @torch.no_grad()
    def prepare_attn_mask(self, mask, query):
        # -- broadcast mask
        assert mask.ndim >= 2 # [..., SeqLen, SeqLen]
        if mask.ndim == 3: # [Batch * Heads, SeqLen, SeqLen]
            mask = mask.view(query.shape)

        return F._canonical_mask(mask, "attn_mask", None, "", query.dtype, False)
    
class SoftmaxValueMultiplication(nn.Module):

    def __init__(self, return_attn=True):
        super().__init__()
        self.return_attn = return_attn
    
    def forward(self, attn_logits, v):
        attention = F.softmax(attn_logits, dim=-1) # [Batch, Head, Query SeqLen, Key SeqLen]
        y = torch.matmul(attention, v) # [Batch, Head, Query SeqLen, Embed]
        if self.return_attn:
            return y, attention
        else:
            return y

class AttentionValueMultiplication(nn.Module):

    def forward(self, attention, v):
        y = torch.matmul(attention, v) # [Batch, Head, Query SeqLen, Embed]
        return y
    
class AttentionValueMultiplicationDetached(nn.Module):

    def forward(self, v, attention=None):
        y = torch.matmul(attention, v) # [Batch, Head, Query SeqLen, Embed]
        return y

class MultiheadAttentionSeparatedLiT(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, batch_first=False, average_attn_weights=False,
                 kdim=None, vdim=None):

        assert embed_dim % num_heads == 0
        assert dropout == 0 # not supported yet
        assert not average_attn_weights # not supported yet

        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        kdim = kdim if kdim is not None else embed_dim
        vdim = vdim if vdim is not None else embed_dim
        self.batch_first = batch_first
        
        self.q_proj = LinearInProjection(embed_dim, embed_dim, bias)
        self.k_proj = LinearInProjection(kdim, embed_dim, bias)
        self.v_proj = LinearInProjection(vdim, embed_dim, bias)

        self.out_proj = LinearOutProjection(embed_dim, embed_dim, bias)

        # LiT modules
        self.q_k_mul = QueryKeyMultiplication()
        self.softmax = nn.Softmax(dim=-1)
        self.attn_value_mul = AttentionValueMultiplication()


    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=True):

        if self.batch_first is False:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, q_seq_length, embed_dim = query.shape
        _, v_seq_length, _ = value.shape
        assert embed_dim == self.embed_dim

        # -- project inputs to new embedding
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # -- reshape for multiheadattention
        q = q.view(batch_size, q_seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, v_seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, v_seq_length, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Embed]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # -- perform attention on each head
        attn_logits = self.q_k_mul(q, k, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        attention = self.softmax(attn_logits)
        y = self.attn_value_mul(attention, v)

        # -- out projection
        y = y.permute(0, 2, 1, 3)
        y = y.reshape(batch_size, q_seq_length, embed_dim)
        out = self.out_proj(y)

        if self.batch_first is False:
            out = out.transpose(0, 1)

        if need_weights:
            return out, attention
        else:
            return out, None



class MultiheadAttentionUnifiedLiT(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, batch_first=True, average_attn_weights=False,
                 kdim=None, vdim=None):

        assert embed_dim % num_heads == 0
        assert batch_first # not supported yet
        assert dropout == 0 # not supported yet
        assert not average_attn_weights # not supported yet

        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        kdim = kdim if kdim is not None else embed_dim
        vdim = vdim if vdim is not None else embed_dim
        
        self.q_proj = LinearInProjection(embed_dim, embed_dim, bias)
        self.k_proj = LinearInProjection(kdim, embed_dim, bias)
        self.v_proj = LinearInProjection(vdim, embed_dim, bias)

        self.out_proj = LinearOutProjection(embed_dim, embed_dim, bias)

        # LiT modules
        self.q_k_mul = QueryKeyMultiplicationOLD()
        self.softmax_v_mul = SoftmaxValueMultiplication()


    def forward(self, query, key, value, mask=None, need_weights=True):

        batch_size, seq_length, embed_dim = query.shape
        assert embed_dim == self.embed_dim

        # -- project inputs to new embedding
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # -- reshape for multiheadattention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Embed]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # -- perform attention on each head
        attn_logits = self.q_k_mul(q, k, mask=mask)
        y, attention = self.softmax_v_mul(attn_logits, v)

        # -- out projection
        y = y.permute(0, 2, 1, 3)
        y = y.reshape(batch_size, seq_length, embed_dim)
        out = self.out_proj(y)

        if need_weights:
            return out, attention
        else:
            return out, None
        

class LinearProjectionLiT(nn.Module):

    def forward(self, x, weight=None, bias=None):
        
        return F.linear(x, weight, bias)
        
class MultiheadAttentionReplacementLiTCORRECT(nn.MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, kdim=None, vdim=None, batch_first=False):

        assert dropout == 0 # not supported yet

        super().__init__(embed_dim, num_heads, dropout=0., bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=False,
                 kdim=kdim, vdim=vdim, batch_first=batch_first, device=None, dtype=None)

        # LiT modules
        self.q_k_mul = QueryKeyMultiplication()
        self.softmax = nn.Softmax(dim=-1)
        self.attn_value_mul = AttentionValueMultiplication()
        self.linear_proj_layer = LinearProjectionLiT()


    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        
        assert not is_causal # not supported yet

        if query is key and key is value:
            self.self_attention = True
        else:
            self.self_attention = False

        if self.batch_first is False:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, q_seq_length, embed_dim = query.shape
        _, v_seq_length, _ = value.shape
        assert embed_dim == self.embed_dim

        # -- project inputs to new embedding
        if not self._qkv_same_embed_dim:
            q = self.linear_proj_layer(query, weight=self.q_proj_weight, bias=None)
            k = self.linear_proj_layer(key, weight=self.k_proj_weight, bias=self.bias_k)
            v = self.linear_proj_layer(value, weight=self.v_proj_weight, bias=self.bias_v)
        elif self.self_attention:
            q, k, v = self.linear_proj_layer(query, weight=self.in_proj_weight, bias=self.in_proj_bias).chunk(3, dim=-1)
        else:
            w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
            b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0) if self.in_proj_bias is not None else (None, None, None)
            q = self.linear_proj_layer(query, weight=w_q, bias=b_q)
            k = self.linear_proj_layer(key, weight=w_k, bias=b_k)
            v = self.linear_proj_layer(value, weight=w_v, bias=b_v)

        # -- reshape for multiheadattention
        q = q.view(batch_size, q_seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, v_seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, v_seq_length, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Embed]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # -- perform attention on each head
        attn_logits = self.q_k_mul(q, k, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        attention = self.softmax(attn_logits)
        y = self.attn_value_mul(attention, v)

        # -- out projection
        y = y.permute(0, 2, 1, 3)
        y = y.reshape(batch_size, q_seq_length, embed_dim)
        out = self.linear_proj_layer(y, weight=self.out_proj.weight, bias=self.out_proj.bias)

        if self.batch_first is False:
            out = out.transpose(0, 1)

        if need_weights:
            if average_attn_weights:
                attention = attention.mean(dim=1)
            return out, attention
        else:
            return out, None
        


class MultiheadAttentionReplacementLiT(nn.MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, kdim=None, vdim=None, batch_first=False):

        assert dropout == 0 # not supported yet

        super().__init__(embed_dim, num_heads, dropout=0., bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=False,
                 kdim=kdim, vdim=vdim, batch_first=batch_first, device=None, dtype=None)

        # LiT modules
        self.q_k_mul = QueryKeyMultiplication()
        self.softmax = nn.Softmax(dim=-1)
        self.attn_value_mul = AttentionValueMultiplication()

        self.linear_q_proj_layer = LinearQueryProjection(1, 1)
        self.linear_v_proj_layer = LinearValueProjection(1, 1)
        self.linear_k_proj_layer = LinearKeyProjection(1, 1)
        self.linear_out_proj_layer = LinearOutProjection(1, 1)


    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        
        assert not is_causal # not supported yet

        if query is key and key is value:
            self.self_attention = True
        else:
            self.self_attention = False

        if self.batch_first is False:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, q_seq_length, embed_dim = query.shape
        _, v_seq_length, _ = value.shape
        assert embed_dim == self.embed_dim

        # -- project inputs to new embedding

        w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0) if self.in_proj_bias is not None else (None, None, None)
        self.linear_q_proj_layer.proj_weight = nn.Parameter(w_q)
        self.linear_q_proj_layer.proj_bias = nn.Parameter(b_q)
        q = self.linear_q_proj_layer(query)
        self.linear_k_proj_layer.proj_weight = nn.Parameter(w_k)
        self.linear_k_proj_layer.proj_bias = nn.Parameter(b_k)
        k = self.linear_k_proj_layer(key)
        self.linear_v_proj_layer.proj_weight = nn.Parameter(w_v)
        self.linear_v_proj_layer.proj_bias = nn.Parameter(b_v)
        v = self.linear_v_proj_layer(value)

        # -- reshape for multiheadattention
        q = q.view(batch_size, q_seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, v_seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, v_seq_length, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Embed]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # -- perform attention on each head
        attn_logits = self.q_k_mul(q, k, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        attention = self.softmax(attn_logits)
        y = self.attn_value_mul(attention, v)

        # -- out projection
        y = y.permute(0, 2, 1, 3)
        y = y.reshape(batch_size, q_seq_length, embed_dim)

        if hasattr(self.out_proj, "weight"):
            self.linear_out_proj_layer.proj_weight = nn.Parameter(self.out_proj.weight)
            self.linear_out_proj_layer.proj_bias = nn.Parameter(self.out_proj.bias)
        else:
            self.linear_out_proj_layer.proj_weight = nn.Parameter(self.out_proj.module.weight)
            self.linear_out_proj_layer.proj_bias = nn.Parameter(self.out_proj.module.bias)

        out = self.linear_out_proj_layer(y)

        if self.batch_first is False:
            out = out.transpose(0, 1)

        if need_weights:
            if average_attn_weights:
                attention = attention.mean(dim=1)
            return out, attention
        else:
            return out, None
        

### --- Reimplementation of the MultiheadAttention module for educational purposes only --- ###    
class MultiheadAttentionScratch(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True):

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = LinearInProjection(embed_dim, embed_dim, bias)
        self.k_proj = LinearInProjection(embed_dim, embed_dim, bias)
        self.v_proj = LinearInProjection(embed_dim, embed_dim, bias)

        self.out_proj = LinearOutProjection(embed_dim, embed_dim, bias)

        raise Warning("This module does not support LiT or Zennit! Just for educational purposes.")


    def forward(self, query, key, value, mask=None):

        batch_size, seq_length, embed_dim = query.shape
        assert embed_dim == self.embed_dim

        # -- project inputs to new embedding
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # -- reshape for multiheadattention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Embed]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # -- perform attention on each head
        d_k = q.shape[-1]
        # matmul computes matrix product on last two dimensions
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) 
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask:
            mask = self.prepare_mask(mask) 
            attn_logits = attn_logits * mask.detach()

        attention = F.softmax(attn_logits, dim=-1) # [Batch, Head, Query SeqLen, Key SeqLen]
        y = torch.matmul(attention, v) # [Batch, Head, Query SeqLen, Embed]

        # -- out projection
        y = y.permute(0, 2, 1, 3)
        y = y.view(batch_size, seq_length, embed_dim)
        out = self.out_proj(y)

        return out, attention
    
    @torch.no_grad()
    def prepare_mask(mask):
        # -- broadcast mask
        assert mask.ndim > 2 # [..., SeqLen, SeqLen]
        if mask.ndim == 3:
            # head dimension missing
            mask = mask.unsqueeze(1)

        # -- set all values == 0 to a large negative value and the remainder to 1
        bool_mask = (mask == 0)
        bool_mask = ~bool_mask + -9e15 * bool_mask
        return bool_mask


### --- Reimplementation of SiwnMultiheadAttention in modularized form so that LiT can be easily applied  --- ###
  
class SiwnLinearProjection(nn.Module):
    """
    nn.Linear reimplemented so that we can apply different zennit rules to the Attention Module
    independet of normal nn.Linear layers.
    """

    def forward(self, x, weight=None, bias=None):
        
        return F.linear(x, weight, bias)

class CosineMultiplication(nn.Module):

    def forward(self, q, k, logit_scale=None):
        """
        ignore normalization and logit_scale in relevance computation by detaching them from the graph
        """
        
        eps = 1e-12
        q = q / torch.clamp_min(torch.norm(q, dim=-1, keepdim=True), eps).detach()
        k = k / torch.clamp_min(torch.norm(k, dim=-1, keepdim=True), eps).detach()
        dot_product = q.matmul(k.transpose(-2, -1))

        if logit_scale is not None:
            scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
            return dot_product * scale.detach()
        else:
            return dot_product

class SwinQueryKeyMultiplication(nn.Module):

    def forward(self, q, k, C=None, num_heads=None):
        q = q * (C // num_heads) ** -0.5
        return q.matmul(k.transpose(-2, -1))

        

class ShiftedWindowAttentionV2LiT(ShiftedWindowAttentionV2):

    def __init__(self, dim: int, window_size, shift_size, num_heads: int, qkv_bias: bool=True, proj_bias: bool=True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__(
            dim,
            window_size,
            shift_size,
            num_heads,
            qkv_bias,
            proj_bias,
            attention_dropout,
            dropout,
        )
  
        self.sum_lit = SumLiT()
        self.linear_proj = SiwnLinearProjection()
        self.cosine = CosineMultiplication()
        self.qk_mul = SwinQueryKeyMultiplication()
        self.softmax = nn.Softmax(dim=-1)
        self.attn_v_mul = AttentionValueMultiplication()


    def forward(self, x):

        ####### ----> GRADIENT = RELEVANCE BACKPROPAGATION #######

        B, H, W, C = x.shape
        # pad feature maps to multiples of window size
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, pad_H, pad_W, _ = x.shape

        self.shift_size = self.shift_size.copy()
        # If window size is larger than feature size, there is no need to shift window
        if self.window_size[0] >= pad_H:
            self.shift_size[0] = 0
        if self.window_size[1] >= pad_W:
            self.shift_size[1] = 0

        # cyclic shift
        if sum(self.shift_size) > 0:
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))

        # partition windows
        num_windows = (pad_H // self.window_size[0]) * (pad_W // self.window_size[1])
        x = x.view(B, pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, self.window_size[0] * self.window_size[1], C)  # B*nW, Ws*Ws, C

        ####### <----- #######

        # multi-head attention
        if self.logit_scale is not None and self.qkv.bias is not None:
            qkv_bias = self.qkv.bias.clone()
            length = qkv_bias.numel() // 3
            qkv_bias[length : 2 * length].zero_()

        qkv = self.linear_proj(x, weight=self.qkv.weight, bias=qkv_bias)

        qkv = qkv.reshape(x.size(0), x.size(1), 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.logit_scale is not None:
            # cosine attention
            attn = self.cosine(q, k, logit_scale=self.logit_scale)
        else:
            attn = self.qk_mul(q, k, C=C, num_heads=self.num_heads)
        
        # add relative position bias
        relative_position_bias = self.get_relative_position_bias()
        attn = self.sum_lit(attn, relative_position_bias.detach())

        if sum(self.shift_size) > 0:
            # generate attention mask
            attn_mask = x.new_zeros((pad_H, pad_W), requires_grad=False)
            h_slices = ((0, -self.window_size[0]), (-self.window_size[0], -self.shift_size[0]), (-self.shift_size[0], None))
            w_slices = ((0, -self.window_size[1]), (-self.window_size[1], -self.shift_size[1]), (-self.shift_size[1], None))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0] : h[1], w[0] : w[1]] = count
                    count += 1
            attn_mask = attn_mask.view(pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1])
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, self.window_size[0] * self.window_size[1])
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

            # mask attention values
            attn = attn.view(x.size(0) // num_windows, num_windows, self.num_heads, x.size(1), x.size(1))
            attn = self.sum_lit(attn, attn_mask.unsqueeze(1).unsqueeze(0).detach())
            attn = attn.view(-1, self.num_heads, x.size(1), x.size(1))

        # compute attention
        attn = self.softmax(attn)
        attn = self.attn_v_mul(attn, v)
        x = attn.transpose(1, 2).reshape(x.size(0), x.size(1), C)

        # project to output
        x = self.linear_proj(x, weight=self.proj.weight, bias=self.proj.bias)

        # reverse windows
        x = x.view(B, pad_H // self.window_size[0], pad_W // self.window_size[1], self.window_size[0], self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

        # reverse cyclic shift
        if sum(self.shift_size) > 0:
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        # unpad features
        x = x[:, :H, :W, :].contiguous()
        return x