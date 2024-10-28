import torch
from zennit.composites import (
    NameMapComposite,
)
from zennit.composites import NameLayerMapComposite
from zennit.rules import Gamma, Epsilon, ZPlus, AlphaBeta, Pass, Norm
from zennit.layer import Sum
from zennit.types import AvgPool, Activation, BatchNorm

from pxp.canonizers import get_vit_canonizer, get_cnn_canonizer

##############################
##############################
######## LiT Packages ########
##############################
##############################
from functools import partial


from pxp.LiT.composites import Composite as LiTComposite
import pxp.LiT.rules as lit_rules

from pxp.LiT.layers import (
    LayerNormTU,
    QueryKeyMultiplication,
    SumLiT,
    AttentionValueMultiplication,
    ConstantMultiplyLiT,
)


def get_integrated_gradient_canonizers():
    return get_vit_canonizer(["ReplaceAttention"])


def get_cnn_composite(model_architecture, configuration):
    composite = NameLayerMapComposite(
        name_map=get_NameMapComposite(model_architecture, configuration),
        layer_map=layer_map_base(),  # there will be a big difference if we don't use this parameter!!
        canonizers=get_cnn_canonizer(model_architecture),
    )

    return composite


def get_vit_composite(model_architecture, configuration):
    root_fn = lambda a, b: (0, 0)
    base_composite = {
        LayerNormTU: lit_rules.EpsilonModule,
        SumLiT: lit_rules.EpsilonModule,
        torch.nn.ReLU: lit_rules.PassModule,
        torch.nn.GELU: lit_rules.PassModule,
        ConstantMultiplyLiT: lit_rules.EpsilonModule,
    }

    AttentionValueMultiplication_rule = (
        lit_rules.EpsilonModule
        if configuration["softmax_rule"] == "CP"
        else partial(
            lit_rules.InputRefXGradientEpsilonModule,
            root_fn=root_fn,
            virtual_bias=False,
        )
    )

    composite = LiTComposite(
        {
            torch.nn.Softmax: get_softmax_rule_by_name(configuration["softmax_rule"]),
            QueryKeyMultiplication: partial(
                lit_rules.InputRefXGradientEpsilonModule,
                root_fn=root_fn,
                virtual_bias=False,
            ),
            AttentionValueMultiplication: AttentionValueMultiplication_rule,
            LayerNormTU: lit_rules.EpsilonModule,
            torch.nn.ReLU: lit_rules.PassModule,
            torch.nn.GELU: lit_rules.PassModule,
            SumLiT: lit_rules.EpsilonModule,
        }
        | base_composite,
        canonizers=get_vit_canonizer(
            ["VitTorchvisionSumCanonizer", "ReplaceAttentionNorm"]
        ),
        zennit_composite=NameMapComposite(
            name_map=get_NameMapComposite(model_architecture, configuration),
        ),
    )

    return composite


def get_NameMapComposite(model_architecture, configuration):
    low_level_layers = configuration["low_level_hidden_layer_rule"]
    mid_level_layers = configuration["mid_level_hidden_layer_rule"]
    high_level_layers = configuration["high_level_hidden_layer_rule"]
    fully_connected_layers = configuration["fully_connected_layers_rule"]
    architecture_composite_wrapper = {
        "vgg16_bn": [
            (
                ["features.0", "features.3", "features.7", "features.10"],
                get_rule_by_name(low_level_layers),
            ),  # low-level convolutions
            (
                [
                    "features.14",
                    "features.17",
                    "features.20",
                    "features.24",
                    "features.27",
                ],
                get_rule_by_name(mid_level_layers),
            ),  # mid-level convolutions
            (
                ["features.30", "features.34", "features.37", "features.40"],
                get_rule_by_name(high_level_layers),
            ),  # high-level convolutions
            (
                ["classifier.0", "classifier.3", "classifier.6"],
                get_rule_by_name(fully_connected_layers),
            ),  # fully connected layers
            (
                [
                    f"features.{i}"
                    for i in [2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42]
                ],
                Pass(),
            ),  # RELUs
            ([f"classifier.{i}" for i in [1, 4]], Pass()),  # RELUs
            (
                [
                    f"features.{i}"
                    for i in [1, 4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]
                ],
                Pass(),
            ),  # BNs
        ],
        "vgg16": [
            (
                ["features.0", "features.2", "features.5", "features.7"],
                get_rule_by_name(low_level_layers),
            ),  # low-level convolutions
            (
                [
                    "features.10",
                    "features.12",
                    "features.14",
                    "features.17",
                    "features.19",
                ],
                get_rule_by_name(mid_level_layers),
            ),  # mid-level convolutions
            (
                ["features.21", "features.24", "features.26", "features.28"],
                get_rule_by_name(high_level_layers),
            ),  # high-level convolutions
            (
                ["classifier.0", "classifier.3", "classifier.6"],
                get_rule_by_name(fully_connected_layers),
            ),  # fully connected layers
            (
                [
                    f"features.{i}"
                    for i in [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29]
                ],
                Pass(),
            ),  # RELUs
            ([f"classifier.{i}" for i in [1, 4]], Pass()),  # RELUs
        ],
        "resnet18": [
            (
                [
                    "conv1",
                    "layer1.0.conv1",
                    "layer1.0.conv2",
                    "layer1.1.conv1",
                    "layer1.1.conv2",
                ],
                get_rule_by_name(low_level_layers),
            ),
            (
                [
                    "layer2.0.conv1",
                    "layer2.0.conv2",
                    "layer2.0.downsample.0",
                    "layer2.1.conv1",
                    "layer2.1.conv2",
                    "layer3.0.conv1",
                    "layer3.0.conv2",
                    "layer3.0.downsample.0",
                    "layer3.1.conv1",
                    "layer3.1.conv2",
                ],
                get_rule_by_name(mid_level_layers),
            ),
            (
                [
                    "layer4.0.conv1",
                    "layer4.0.conv2",
                    "layer4.0.downsample.0",
                    "layer4.1.conv1",
                    "layer4.1.conv2",
                ],
                get_rule_by_name(high_level_layers),
            ),
            (["relu"], Pass()),
            (
                [f"layer{i}.{j}.relu" for i in [1, 2, 3, 4] for j in [0, 1]],
                Pass(),
            ),  # RELUs
            (["fc"], get_rule_by_name(fully_connected_layers)),
        ],
        "resnet50": [
            (
                [
                    "conv1",
                    "layer1.0.conv1",
                    "layer1.0.conv2",
                    "layer1.0.conv3",
                    "layer1.0.downsample.0",
                    "layer1.1.conv1",
                    "layer1.1.conv2",
                    "layer1.1.conv3",
                    "layer1.2.conv1",
                    "layer1.2.conv2",
                    "layer1.2.conv3",
                ],
                get_rule_by_name(low_level_layers),
            ),
            (
                [
                    "layer2.0.conv1",
                    "layer2.0.conv2",
                    "layer2.0.conv3",
                    "layer2.0.downsample.0",
                    "layer2.1.conv1",
                    "layer2.1.conv2",
                    "layer2.1.conv3",
                    "layer2.2.conv1",
                    "layer2.2.conv2",
                    "layer2.2.conv3",
                    "layer2.3.conv1",
                    "layer2.3.conv2",
                    "layer2.3.conv3",
                    "layer3.0.conv1",
                    "layer3.0.conv2",
                    "layer3.0.conv3",
                    "layer3.0.downsample.0",
                    "layer3.1.conv1",
                    "layer3.1.conv2",
                    "layer3.1.conv3",
                    "layer3.2.conv1",
                    "layer3.2.conv2",
                    "layer3.2.conv3",
                    "layer3.3.conv1",
                    "layer3.3.conv2",
                    "layer3.3.conv3",
                    "layer3.4.conv1",
                    "layer3.4.conv2",
                    "layer3.4.conv3",
                    "layer3.5.conv1",
                    "layer3.5.conv2",
                    "layer3.5.conv3",
                ],
                get_rule_by_name(mid_level_layers),
            ),
            (
                [
                    "layer4.0.conv1",
                    "layer4.0.conv2",
                    "layer4.0.conv3",
                    "layer4.0.downsample.0",
                    "layer4.1.conv1",
                    "layer4.1.conv2",
                    "layer4.1.conv3",
                    "layer4.2.conv1",
                    "layer4.2.conv2",
                    "layer4.2.conv3",
                ],
                get_rule_by_name(high_level_layers),
            ),
            (["relu"], Pass()),
            (
                [f"layer{i}.{j}.relu" for i in [1, 4] for j in [0, 1, 2]],
                Pass(),
            ),  # RELUs
            ([f"layer2.{j}.relu" for j in [0, 1, 2, 3]], Pass()),  # RELUs
            ([f"layer3.{j}.relu" for j in [0, 1, 2, 3, 4, 5]], Pass()),  # RELUs
            (["fc"], get_rule_by_name(fully_connected_layers)),
        ],
        "vit_b_16": [
            (
                [
                    f"encoder.layers.encoder_layer_{i}.mlp.0"
                    for i in [
                        0,
                        1,
                        2,
                    ]
                ],
                get_rule_by_name(low_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.mlp.3"
                    for i in [
                        0,
                        1,
                        2,
                    ]
                ],
                get_rule_by_name(low_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.self_attention.q_proj"
                    for i in [
                        0,
                        1,
                        2,
                    ]
                ],
                get_rule_by_name(low_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.self_attention.k_proj"
                    for i in [
                        0,
                        1,
                        2,
                    ]
                ],
                get_rule_by_name(low_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.self_attention.v_proj"
                    for i in [
                        0,
                        1,
                        2,
                    ]
                ],
                get_rule_by_name(low_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.self_attention.out_proj"
                    for i in [
                        0,
                        1,
                        2,
                    ]
                ],
                get_rule_by_name(low_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.mlp.0"
                    for i in [
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                    ]
                ],
                get_rule_by_name(mid_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.mlp.3"
                    for i in [
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                    ]
                ],
                get_rule_by_name(mid_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.self_attention.q_proj"
                    for i in [
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                    ]
                ],
                get_rule_by_name(mid_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.self_attention.k_proj"
                    for i in [
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                    ]
                ],
                get_rule_by_name(mid_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.self_attention.v_proj"
                    for i in [
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                    ]
                ],
                get_rule_by_name(mid_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.self_attention.out_proj"
                    for i in [
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                    ]
                ],
                get_rule_by_name(mid_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.mlp.0"
                    for i in [
                        9,
                        10,
                        11,
                    ]
                ],
                get_rule_by_name(high_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.mlp.3"
                    for i in [
                        9,
                        10,
                        11,
                    ]
                ],
                get_rule_by_name(high_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.self_attention.q_proj"
                    for i in [
                        9,
                        10,
                        11,
                    ]
                ],
                get_rule_by_name(high_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.self_attention.k_proj"
                    for i in [
                        9,
                        10,
                        11,
                    ]
                ],
                get_rule_by_name(high_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.self_attention.v_proj"
                    for i in [
                        9,
                        10,
                        11,
                    ]
                ],
                get_rule_by_name(high_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.self_attention.out_proj"
                    for i in [
                        9,
                        10,
                        11,
                    ]
                ],
                get_rule_by_name(high_level_layers),
            ),
            (["heads.head"], get_rule_by_name(fully_connected_layers)),
        ],
        "vit_l_16": [
            (
                [f"encoder.layers.encoder_layer_{i}.mlp.0" for i in [0, 1, 2, 3, 4, 5]],
                get_rule_by_name(low_level_layers),
            ),
            (
                [f"encoder.layers.encoder_layer_{i}.mlp.3" for i in [0, 1, 2, 3, 4, 5]],
                get_rule_by_name(low_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.mlp.0"
                    for i in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
                ],
                get_rule_by_name(mid_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.mlp.3"
                    for i in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
                ],
                get_rule_by_name(mid_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.mlp.0"
                    for i in [18, 19, 20, 21, 22, 23]
                ],
                get_rule_by_name(high_level_layers),
            ),
            (
                [
                    f"encoder.layers.encoder_layer_{i}.mlp.3"
                    for i in [18, 19, 20, 21, 22, 23]
                ],
                get_rule_by_name(high_level_layers),
            ),
            (["heads.head"], get_rule_by_name(fully_connected_layers)),
        ],
    }

    return architecture_composite_wrapper[model_architecture]


def get_softmax_rule_by_name(rule):
    softmax_fn = lambda a: (0,)
    softmax_rule_wrapper = {
        "Epsilon": lit_rules.IxGSoftmaxBriefModule,
        "ZPlus_vbias_T": partial(
            lit_rules.InputRefXGSoftmaxZPlusModuleVMAP,
            virtual_bias=True,
            ref_fn=softmax_fn,
        ),
        "ZPlus_vbias_F": partial(
            lit_rules.InputRefXGSoftmaxZPlusModuleVMAP,
            virtual_bias=False,
            ref_fn=softmax_fn,
        ),
        "CP": lit_rules.BlockModule,
    }
    return softmax_rule_wrapper[rule]


# def get_rule_by_name(rule, rule_config):
def get_rule_by_name(rule):
    rules_wrapper = {
        "Epsilon": Epsilon,
        "Gamma": Gamma,
        "ZPlus": ZPlus,
        "AlphaBeta": AlphaBeta,
    }
    if rule in ["ZPlus", "Epsilon"]:
        return rules_wrapper[rule]()
    elif rule == "Gamma":
        # return rules_wrapper[rule](rule_config)
        rules_wrapper[rule]()
    elif rule == "AlphaBeta":
        return rules_wrapper[rule](2, 1)


def layer_map_base(stabilizer=1e-6):
    """Return a basic layer map (list of 2-tuples) shared by all built-in LayerMapComposites.

    Parameters
    ----------
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.

    Returns
    -------
    list[tuple[tuple[torch.nn.Module, ...], Hook]]
        Basic ayer map shared by all built-in LayerMapComposites.
    """
    return [
        (Activation, Pass()),
        (Sum, Norm()),
        (AvgPool, Norm()),
        (BatchNorm, Pass()),
    ]
