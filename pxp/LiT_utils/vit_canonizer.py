import torch
from torchvision.models.vision_transformer import EncoderBlock, Encoder
from pxp.LiT.layers import SumLiT, MultiheadAttentionSeparatedLiT, LayerNormTU
from pxp.LiT.canonizers import CanonizerReplaceAttribute, CanonizerReplaceModule


class VitTorchvisionSumCanonizer(CanonizerReplaceAttribute):
    """Canonizer specifically for Encoder and EncoderBlock of torchvision.models.vit* type models."""

    def __init__(self):

        attr_map = {
            EncoderBlock: {
                "sum_layer_attn": self.get_sum_layer,
                "sum_layer_mlp": self.get_sum_layer,
                "forward": self.get_forward_block,
            },
            Encoder: {
                "sum_layer": self.get_sum_layer,
                "forward": self.get_forward_encoder,
            },
        }

        super().__init__(attr_map)

    def get_sum_layer(self, _module):
        return SumLiT()

    def get_forward_block(self, module):
        return self.forward_block.__get__(module)

    def get_forward_encoder(self, module):
        return self.forward_encoder.__get__(module)

    @staticmethod
    def forward_block(self, input: torch.Tensor):

        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        # x = x + input
        x = self.sum_layer_attn(x, input)

        y = self.ln_2(x)
        y = self.mlp(y)
        # return x + y
        return self.sum_layer_mlp(x, y)

    @staticmethod
    def forward_encoder(self, input: torch.Tensor):

        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        # input = input + self.pos_embedding
        input = self.sum_layer(input, self.pos_embedding)
        return self.ln(self.layers(self.dropout(input)))


class ReplaceAttentionNorm(CanonizerReplaceModule):
    # TODO: bug when removed

    def __init__(self, attn_type=MultiheadAttentionSeparatedLiT, norm_type=LayerNormTU):

        self.layer_map = {
            torch.nn.MultiheadAttention: self.create_attention_layer,
            torch.nn.LayerNorm: self.create_norm_layer,
        }

        self.attn_type = attn_type
        self.norm_type = norm_type

        super().__init__(self.layer_map)

    def create_norm_layer(self, source):

        # -- LayerNorm
        norm_layer = self.norm_type(source.normalized_shape, source.eps)

        for name, param in source.named_parameters():

            norm_layer.register_parameter(name, param)

        return norm_layer

    def create_attention_layer(self, source):

        # -- MultiheadAttention
        if source.in_proj_bias is None:
            bias = False
        else:
            bias = True

        dest_layer = self.attn_type(
            source.embed_dim,
            source.num_heads,
            bias=bias,
            batch_first=source.batch_first,
            kdim=source.kdim,
            vdim=source.vdim,
        )

        if not source._qkv_same_embed_dim:
            dest_layer.q_proj.proj_weight.data.copy_(source.q_proj_weight.data)
            dest_layer.k_proj.proj_weight.data.copy_(source.k_proj_weight.data)
            dest_layer.v_proj.proj_weight.data.copy_(source.v_proj_weight.data)
        else:
            dest_layer.q_proj.proj_weight.data.copy_(
                source.in_proj_weight.data[: source.embed_dim]
            )
            dest_layer.k_proj.proj_weight.data.copy_(
                source.in_proj_weight.data[source.embed_dim : source.embed_dim * 2]
            )
            dest_layer.v_proj.proj_weight.data.copy_(
                source.in_proj_weight.data[source.embed_dim * 2 : source.embed_dim * 3]
            )

        dest_layer.out_proj.proj_weight.data.copy_(source.out_proj.weight.data)

        if source.in_proj_bias is not None:
            dest_layer.q_proj.proj_bias.data.copy_(
                source.in_proj_bias.data[: source.embed_dim]
            )
            dest_layer.k_proj.proj_bias.data.copy_(
                source.in_proj_bias.data[source.embed_dim : source.embed_dim * 2]
            )
            dest_layer.v_proj.proj_bias.data.copy_(
                source.in_proj_bias.data[source.embed_dim * 2 : source.embed_dim * 3]
            )
            dest_layer.out_proj.proj_bias.data.copy_(source.out_proj.bias.data)

        dest_layer.to(source.out_proj.weight.device)

        return dest_layer


class ReplaceAttention(CanonizerReplaceModule):

    def __init__(self, attn_type=MultiheadAttentionSeparatedLiT):

        self.layer_map = {
            torch.nn.MultiheadAttention: self.create_attention_layer,
        }

        self.attn_type = attn_type

        super().__init__(self.layer_map)

    def create_attention_layer(self, source):

        # -- MultiheadAttention
        if source.in_proj_bias is None:
            bias = False
        else:
            bias = True

        dest_layer = self.attn_type(
            source.embed_dim,
            source.num_heads,
            bias=bias,
            batch_first=source.batch_first,
            kdim=source.kdim,
            vdim=source.vdim,
        )

        if not source._qkv_same_embed_dim:
            dest_layer.q_proj.proj_weight.data.copy_(source.q_proj_weight.data)
            dest_layer.k_proj.proj_weight.data.copy_(source.k_proj_weight.data)
            dest_layer.v_proj.proj_weight.data.copy_(source.v_proj_weight.data)
        else:
            dest_layer.q_proj.proj_weight.data.copy_(
                source.in_proj_weight.data[: source.embed_dim]
            )
            dest_layer.k_proj.proj_weight.data.copy_(
                source.in_proj_weight.data[source.embed_dim : source.embed_dim * 2]
            )
            dest_layer.v_proj.proj_weight.data.copy_(
                source.in_proj_weight.data[source.embed_dim * 2 : source.embed_dim * 3]
            )

        dest_layer.out_proj.proj_weight.data.copy_(source.out_proj.weight.data)

        if source.in_proj_bias is not None:
            dest_layer.q_proj.proj_bias.data.copy_(
                source.in_proj_bias.data[: source.embed_dim]
            )
            dest_layer.k_proj.proj_bias.data.copy_(
                source.in_proj_bias.data[source.embed_dim : source.embed_dim * 2]
            )
            dest_layer.v_proj.proj_bias.data.copy_(
                source.in_proj_bias.data[source.embed_dim * 2 : source.embed_dim * 3]
            )
            dest_layer.out_proj.proj_bias.data.copy_(source.out_proj.bias.data)

        dest_layer.to(source.out_proj.weight.device)

        return dest_layer


class ReplaceNorm(CanonizerReplaceModule):

    def __init__(self, norm_type=LayerNormTU):

        self.layer_map = {
            torch.nn.LayerNorm: self.create_norm_layer,
        }

        self.norm_type = norm_type

        super().__init__(self.layer_map)

    def create_norm_layer(self, source):

        # -- LayerNorm
        norm_layer = self.norm_type(source.normalized_shape, source.eps)

        for name, param in source.named_parameters():

            norm_layer.register_parameter(name, param)

        return norm_layer
